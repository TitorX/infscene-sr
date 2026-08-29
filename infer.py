"""InfScene-SR: whole-scene super-resolution by joint denoising of overlapping tiles.

Every reverse diffusion step advances all tiles of the scene by one step and
fuses their overlaps before the next step. Fusion follows the spatially
decoupled variance correction (SDVC), so each tile's contribution is computed
from tile-local quantities and precomputed geometry maps, and the fused scene is
a plain sum of contributions. Tiles can therefore be processed by independent
workers on different GPUs, and the scene itself never has to fit in GPU memory.
"""
import argparse
import concurrent.futures
import logging
import os
import signal
from typing import Optional, Tuple

import h5py
import numpy as np
import yaml
from PIL import Image
from osgeo import gdal; gdal.UseExceptions()
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr3 import SR3LightningModule


torch.set_float32_matmul_precision('medium')


def get_views(image_size: Tuple[int, int],
              patch_size: Tuple[int, int],
              stride: Tuple[int, int]) -> np.ndarray:
    """Tile a canvas with overlapping views.

    Returns an (N, 4) array of (start_h, start_w, end_h, end_w). A regular grid
    of `stride` covers the canvas; when the grid leaves a strip uncovered at the
    right or bottom edge, extra views flush with that edge are appended.
    """
    H, W = image_size
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    if H < patch_h or W < patch_w:
        raise ValueError(
            f"Scene {H}x{W} is smaller than the tile {patch_h}x{patch_w}")

    n_h = (H - patch_h) // stride_h + 1
    n_w = (W - patch_w) // stride_w + 1

    views = [(i * stride_h, j * stride_w, i * stride_h + patch_h, j * stride_w + patch_w)
             for i in range(n_h) for j in range(n_w)]

    # Strips the regular grid does not reach.
    rows_left = (n_h - 1) * stride_h + patch_h < H
    cols_left = (n_w - 1) * stride_w + patch_w < W

    if rows_left:
        start_h = H - patch_h
        views += [(start_h, j * stride_w, H, j * stride_w + patch_w)
                  for j in range(n_w)]
    if cols_left:
        start_w = W - patch_w
        views += [(i * stride_h, start_w, i * stride_h + patch_h, W)
                  for i in range(n_h)]
    if rows_left and cols_left:
        views.append((H - patch_h, W - patch_w, H, W))

    return np.array(views)


def generate_guidance_map(size: int) -> torch.Tensor:
    """Per-tile fusion weight w, peaking at the tile center.

    The weight is one minus the normalized distance to the center, plus a small
    constant so that corner pixels never get exactly zero weight.
    """
    center = size // 2
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    max_distance = torch.sqrt(torch.tensor(2 * (center ** 2)))
    weight = 1 - (distance / max_distance) + 1e-4
    return weight.unsqueeze(0)


class LargeImage:
    """The scene being denoised, backed by HDF5 files in `path`.

    Three arrays live on disk instead of in RAM. `latent.h5` holds the current
    scene state y_t and the timestep it belongs to, so an interrupted run
    resumes from the last saved step. `meta.h5` holds the tile weight w and the
    two geometry maps of SDVC, W = sum_j w_j and gmap2 = sum_j w_j^2, from which
    S = sqrt(gmap2); both depend only on the tiling, so they are built once.

    Indexing the object yields one tile of everything a worker needs for a step.
    """

    def __init__(self,
                 sr_image_path: str,
                 stride: Tuple[int, int],
                 patch_shape: Tuple[int, int, int],  # (C, H, W)
                 path: str,
                 bands: Optional[list] = None,
                 pin_memory: bool = False,
                 norm_range: Optional[Tuple[float, float]] = None):
        self.sr_image_path = sr_image_path

        ds = gdal.Open(self.sr_image_path)
        self.sr_image = ds.ReadAsArray()
        del ds
        if bands is not None:
            self.sr_image = self.sr_image[bands]
        self.sr_image = self.sr_image.astype(np.float32)

        # Min-max normalize to [0, 1], then to [-1, 1]. norm_range fixes the
        # scale explicitly, so that a crop of a scene is normalized exactly as
        # the whole scene is (the crop's own extrema would stretch it).
        lo, hi = norm_range if norm_range is not None else \
            (self.sr_image.min(), self.sr_image.max())
        self.sr_image = (self.sr_image - lo) / (hi - lo) * 2 - 1

        self.image_size = self.sr_image.shape[1:]
        self.stride = stride
        self.patch_shape = patch_shape
        self.path = path
        self.pin_memory = pin_memory
        self.views = get_views(self.image_size, patch_shape[1:], stride)

        self.meta_path = f"{path}/meta.h5"
        self.latent_path = f"{path}/latent.h5"

        if not os.path.exists(self.meta_path):
            self._build_meta()
        if not os.path.exists(self.latent_path):
            self._build_latent()
        else:
            with h5py.File(self.latent_path, 'r') as latent_file:
                self.timestep = latent_file['timestep'][()]

        self._latent_memory = None
        self._meta_memory = None
        if self.pin_memory:
            with h5py.File(self.latent_path, 'r') as latent_file:
                self._latent_memory = latent_file['latent'][:]
            # The geometry maps are the same at every timestep, so keep them in
            # RAM instead of re-decompressing the gzip'd chunks 2000 times.
            with h5py.File(self.meta_path, 'r') as meta_file:
                self._meta_memory = (meta_file['gmap'][:],
                                     meta_file['gmap2'][:],
                                     meta_file['g'][:])

        # Accumulator for the contributions of the step in progress.
        self._latent = np.zeros((patch_shape[0], *self.image_size), dtype=np.float32)

    def _build_meta(self):
        # 512 MB chunk cache, since overlapping tiles revisit the same chunks.
        with h5py.File(self.meta_path, 'w', rdcc_nbytes=512 * 1024 * 1024) as meta_file:
            g = generate_guidance_map(self.patch_shape[1]).numpy()
            meta_file.create_dataset('g', data=g, dtype=np.float32)
            g2 = g ** 2

            def empty_map(name):
                return meta_file.create_dataset(
                    name, shape=(self.patch_shape[0], *self.image_size),
                    dtype=np.float32, compression="gzip", fillvalue=0,
                    chunks=self.patch_shape)

            gmap, gmap2 = empty_map('gmap'), empty_map('gmap2')
            for start_h, start_w, end_h, end_w in tqdm(self.views,
                                                       desc="Building geometry maps"):
                gmap[:, start_h:end_h, start_w:end_w] += g
                gmap2[:, start_h:end_h, start_w:end_w] += g2

    def _build_latent(self):
        with h5py.File(self.latent_path, 'w') as latent_file:
            self.timestep = -1
            latent_file.create_dataset('timestep', data=self.timestep)
            latent = latent_file.create_dataset(
                'latent', shape=(self.patch_shape[0], *self.image_size),
                dtype=np.float32, compression="gzip", chunks=self.patch_shape)

            # Fill chunk by chunk so that every chunk is written exactly once.
            C, Ph, Pw = self.patch_shape
            H, W = self.image_size
            for h in tqdm(range(0, H, Ph), desc="Initializing latent"):
                for w in range(0, W, Pw):
                    curr_h, curr_w = min(Ph, H - h), min(Pw, W - w)
                    latent[:, h:h + curr_h, w:w + curr_w] = \
                        torch.randn((C, curr_h, curr_w)).numpy()

    def _read_latent(self, start_h, start_w, end_h, end_w):
        if self._latent_memory is not None:
            return self._latent_memory[:, start_h:end_h, start_w:end_w]
        with h5py.File(self.latent_path, 'r') as latent_file:
            return latent_file['latent'][:, start_h:end_h, start_w:end_w]

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        start_h, start_w, end_h, end_w = self.views[index]
        latent = self._read_latent(start_h, start_w, end_h, end_w)
        condition = self.sr_image[:, start_h:end_h, start_w:end_w]

        if self._meta_memory is not None:
            gmap, gmap2, g = self._meta_memory
            return (index, latent, condition,
                    gmap[:, start_h:end_h, start_w:end_w],
                    gmap2[:, start_h:end_h, start_w:end_w], g)

        with h5py.File(self.meta_path, 'r') as meta_file:
            return (index, latent, condition,
                    meta_file['gmap'][:, start_h:end_h, start_w:end_w],
                    meta_file['gmap2'][:, start_h:end_h, start_w:end_w],
                    meta_file['g'][:])

    def update(self, index, value):
        """Add the contributions of a batch of tiles to the step accumulator."""
        for i, v in zip(index, value):
            start_h, start_w, end_h, end_w = self.views[i]
            self._latent[:, start_h:end_h, start_w:end_w] += v

    def apply(self, timestep, save_to_disk=True):
        """Close a step: the accumulated contributions become the new state."""
        self.timestep = timestep

        if not self.pin_memory or save_to_disk:
            tmp_file = self.latent_path + '.tmp'
            with h5py.File(tmp_file, 'w') as latent_file:
                latent_file.create_dataset('latent', data=self._latent)
                latent_file.create_dataset('timestep', data=self.timestep)
            os.rename(tmp_file, self.latent_path)

        if self.pin_memory:
            self._latent_memory = self._latent

        self._latent = np.zeros_like(self._latent)

    def to_tiff(self, output_path, block_rows=1024):
        """Write the current state as a georeferenced 8-bit GeoTIFF.

        The scene is converted and written in horizontal slabs, so a scene far
        larger than RAM can still be written out.
        """
        ori_ds = gdal.Open(self.sr_image_path)
        geotransform = ori_ds.GetGeoTransform()
        projection = ori_ds.GetProjection()
        del ori_ds

        C = self.patch_shape[0]
        H, W = self.image_size

        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_path, W, H, C, gdal.GDT_Byte,
                           options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)

        for y in range(0, H, block_rows):
            rows = min(block_rows, H - y)
            slab = self._read_latent(y, 0, y + rows, W)
            slab = ((np.clip(slab, -1, 1) + 1) / 2 * 255.0).round().astype(np.uint8)
            for i in range(C):
                ds.GetRasterBand(i + 1).WriteArray(slab[i], 0, y)

        ds.FlushCache()
        del ds

    @staticmethod
    def to_preview(tif_path, png_path, max_size=2048):
        """Downscaled RGB PNG of a written GeoTIFF, to watch a run progress."""
        ds = gdal.Open(tif_path)
        scale = max(1.0, max(ds.RasterXSize, ds.RasterYSize) / max_size)
        preview = ds.ReadAsArray(buf_xsize=round(ds.RasterXSize / scale),
                                 buf_ysize=round(ds.RasterYSize / scale))
        del ds

        if preview.ndim == 3:
            preview = np.transpose(preview[:3], (1, 2, 0)).squeeze()
        Image.fromarray(preview).save(png_path)


def create_model(opt, device, checkpoint_path):
    model = SR3LightningModule(len(opt['dataset']['bands']),
                               opt['dataset']['patch_size'])
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.to(device)
    model.eval()
    return model


class LargeImageSRPipeline:
    """One reverse step for one batch of tiles, returning fused contributions."""

    def __init__(self, model, opt, device, amp=False, fusion_mode='sdvc'):
        self.opt = opt
        self.device = device
        self.model = model
        self.amp = amp
        self.fusion_mode = fusion_mode

    @torch.no_grad()
    def __call__(self, batch, t: torch.Tensor):
        batch = [batch[0]] + [i.to(self.device) for i in batch[1:]]
        index, patch, condition, gmap, gmap2, g = batch

        with torch.amp.autocast('cuda', enabled=self.amp):
            mu, variance = self.model.p_sample_intermediate(
                patch, t, condition_x=condition)

        # y = mu + sigma * eps, the tile's own sampled state.
        y = mu + variance

        if self.fusion_mode == 'avg':
            # Naive joint denoising: guidance-weighted average of the sampled
            # states. Independent per-tile noise partly cancels in overlaps,
            # which erodes the sampling variance.
            output = (g / gmap) * y
        else:
            # SDVC: with W = sum_j w_j and S = sqrt(sum_j w_j^2), tile i
            # contributes (w/S) y + (1 - W/S) (w/W) mu. Summing the
            # contributions of all tiles restores variance exactly, and no tile
            # needs to see another tile's prediction.
            s = gmap2 ** 0.5
            output = (g / s) * y + ((1 - gmap / s) * g / gmap) * mu

        return index, output


def inference(in_queue, out_queue, device, model_opt, checkpoint_path, amp,
              fusion_mode):
    """GPU worker: one process per device, fed tiles through `in_queue`."""
    model = create_model(model_opt, device, checkpoint_path)
    pipeline = LargeImageSRPipeline(model, model_opt, device, amp=amp,
                                    fusion_mode=fusion_mode)
    try:
        while True:
            t, batch = in_queue.get()
            if t == -1:
                return
            out_queue.put(pipeline(batch, t))
    except KeyboardInterrupt:
        return  # the parent is shutting the run down


def feed_data(in_queue, dataloader, t):
    # Both helpers run in threads, where an interrupt would otherwise surface as
    # a traceback from wherever the thread happened to be.
    try:
        for batch in dataloader:
            in_queue.put((t, batch))
    except KeyboardInterrupt:
        return


def write_results(in_queue, out_queue, dataset, num_results):
    pbar = tqdm(total=num_results, leave=False)
    try:
        for _ in range(num_results):
            index, contribution = out_queue.get()
            dataset.update(index.cpu(), contribution.cpu().numpy())
            pbar.set_description(
                f"QOut: {out_queue.qsize()}, QIn: {in_queue.qsize()}")
            pbar.update(1)
    except KeyboardInterrupt:
        return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Super-resolve a whole scene by joint denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml',
                        help='YAML config of the trained model (bands, patch size)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Trained SR3 checkpoint')
    parser.add_argument('--sr-path', type=str, required=True,
                        help='Input raster, the LR observation upsampled to the '
                             'HR grid (see scripts/create_sr_from_hr.py)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory, also used to resume a run')
    parser.add_argument('--stride', type=int, default=384,
                        help='Tile stride in pixels; the tile size is the '
                             'training patch size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Tiles per forward pass, per GPU worker')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers, restarted at every timestep; '
                             'raise it only when the scene state is read from '
                             'disk, that is without --pin-memory')
    parser.add_argument('--queue-size', type=int, default=100,
                        help='Tile batches buffered between the loader and the '
                             'GPU workers')
    parser.add_argument('--save-freq', type=int, default=200,
                        help='Write the latent, a GeoTIFF and a preview every '
                             'this many timesteps')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Keep the scene state and geometry maps in RAM')
    parser.add_argument('--amp', action='store_true',
                        help='Run the U-Net under bf16/fp16 autocast')
    parser.add_argument('--fusion-mode', type=str, default='sdvc',
                        choices=['sdvc', 'avg'],
                        help='sdvc is the variance-corrected fusion of the '
                             'paper, avg the naive weighted averaging baseline')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed of the initial scene noise')
    parser.add_argument('--norm-range', type=str, default=None,
                        help='"lo,hi" to normalize the input with instead of '
                             'its own min/max; pass 0,255 whenever the input is '
                             'a crop of a larger scene, so that the crop is '
                             'scaled the way the whole scene would be')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    with open(args.config, 'r') as f:
        model_opt = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    patch_shape = (len(model_opt['dataset']['bands']),
                   model_opt['dataset']['patch_size'],
                   model_opt['dataset']['patch_size'])
    # Config bands are 1-indexed, GDAL array indices are 0-indexed.
    bands = [int(b) - 1 for b in model_opt['dataset']['bands']]

    dataset = LargeImage(
        sr_image_path=args.sr_path,
        stride=(args.stride, args.stride),
        patch_shape=patch_shape,
        path=args.output,
        bands=bands,
        pin_memory=args.pin_memory,
        norm_range=(tuple(float(v) for v in args.norm_range.split(','))
                    if args.norm_range else None),
    )
    logger.info(f"Scene {dataset.image_size}, {len(dataset)} tiles")

    current_timestep = 2000 if dataset.timestep == -1 else dataset.timestep
    if dataset.timestep != -1:
        logger.info(f"Resuming from timestep {current_timestep}")
    timesteps = [t for t in reversed(range(0, 2000)) if t < current_timestep]

    devices = torch.cuda.device_count()
    logger.info(f"Number of devices: {devices}")

    # CUDA cannot be initialized in a forked child, so the GPU workers must be
    # spawned. Using an explicit spawn context instead of setting the global
    # start method keeps the DataLoader workers on the cheaper fork.
    ctx = mp.get_context('spawn')
    in_queue = ctx.Queue(maxsize=args.queue_size)
    out_queue = ctx.Queue(maxsize=args.queue_size)

    processes = []
    for i in range(devices):
        p = ctx.Process(
            target=inference,
            args=(in_queue, out_queue, torch.device(f'cuda:{i}'), model_opt,
                  args.checkpoint, args.amp, args.fusion_mode))
        p.start()
        processes.append(p)

    # persistent_workers stays off on purpose: the workers are forked again at
    # every timestep, which is how they see the scene state written by the
    # previous step.
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    try:
        # Driven by hand rather than by iteration, so that an interrupt does
        # not leave an abandoned generator to complain about at collection.
        progress = tqdm(total=len(timesteps))
        for t in timesteps:
            progress.set_description(desc=f"Timestep {t}")
            progress.update(1)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(feed_data, in_queue, dataloader, t)
                executor.submit(write_results, in_queue, out_queue, dataset,
                                len(dataloader))

            save = t % args.save_freq == 0
            dataset.apply(t, save_to_disk=save)
            if save:
                dataset.to_tiff(f"{args.output}/image.tif")
                dataset.to_preview(f"{args.output}/image.tif",
                                   f"{args.output}/preview_{t}.png")

        dataset.to_tiff(f"{args.output}/image.tif")
        dataset.to_preview(f"{args.output}/image.tif", f"{args.output}/preview.png")

        for _ in range(devices):
            in_queue.put((-1, None))
        for p in processes:
            p.join()
        logger.info("Inference completed")
    except KeyboardInterrupt:
        # An interrupt reaches every process in the group and is forwarded again
        # by the launcher, so stop listening before cleaning up. The GPU workers
        # block on the queue and the collector threads block on their results,
        # so nothing unwinds on its own; kill the workers and leave without
        # waiting for the threads.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.info("Interrupted. Rerun the same command to continue from the "
                    "last saved timestep.")
        for p in processes:
            p.terminate()
        for q in (in_queue, out_queue):
            q.cancel_join_thread()
        os._exit(130)
