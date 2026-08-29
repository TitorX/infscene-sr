# InfScene-SR

Code for the paper

**InfScene-SR: Seamless Super-Resolution of Arbitrarily Large Remote-Sensing Scenes via Variance-Preserving Joint Denoising**

Shoukun Sun, Zhe Wang, Xiang Que, Jiyin Zhang, Xiaogang Ma

[arXiv:2602.19736](https://arxiv.org/abs/2602.19736)

InfScene-SR super-resolves a scene far larger than the crop its diffusion model was
trained on. It covers the scene with overlapping tiles, advances every tile by one
reverse diffusion step, and fuses the overlaps at each step with a variance-corrected
rule whose contributions are tile-local. Independent workers therefore process tiles
across GPUs, and the scene never has to fit in GPU memory.

This repository holds the training and inference code only. The evaluation protocol
and the downstream segmentation experiments of the paper are not included.

```
train.py                       train the SR3 diffusion model on patches
infer.py                       super-resolve a whole scene by joint denoising
dataset.py                     paired patches extracted from large rasters
lf_project.py                  optional post-process, see below
sr3/                           the SR3 model and its U-Net
config/config.yaml             the configuration used in the paper
config/demo.yaml               small configuration for the smoke test
scripts/create_sr_from_hr.py   build the inference input from an HR raster
scripts/make_demo_data.py      synthetic scene for the smoke test
```

Run every command from the repository root. The scripts import each other flatly and
resolve `config/`, `data/` and `runs/` relative to the working directory.

## Setup

Dependencies are managed with [pixi](https://pixi.sh). GDAL is the reason for a conda
environment rather than plain pip, so install it this way.

```bash
pixi install
pixi shell          # or prefix every command below with `pixi run`
```

NAIP is distributed as MrSID (`.sid`), which GDAL reads only through a plugin that
cannot be shipped prebuilt. The `gdal-mrsid-builder` package provides a `build-mrsid`
command that downloads the MrSID SDK and compiles the plugin against the GDAL in your
environment, so building it needs network access and a few minutes.

```bash
pixi run build-mrsid
pixi run gdalinfo --formats | grep MrSID   # prints the driver once the plugin is in place
```

Everything else works on GeoTIFFs without this step, and you can then drop
`gdal-mrsid-builder` from `pixi.toml` in favour of `gdal` from conda-forge.

Training and inference need an NVIDIA GPU. Both use every visible GPU, so set
`CUDA_VISIBLE_DEVICES` to restrict a run to some of them. The manifest pins Python
3.12 and a CUDA 12 build of PyTorch, and the installed environment is about 10 GB.
The paper was produced on one workstation with two 24 GB L4 GPUs.

## Data

Training needs high-resolution imagery and two polygon shapefiles, one per split. A
third, for a test split, is optional.

```
data/
├── images/          # .tif, .tiff or .sid, searched recursively
└── masks/
    ├── train.shp    # + .shx .dbf .prj
    ├── val.shp
    └── test.shp    # optional, set `masks.test` to null to skip it
```

The masks mark which areas of the imagery each split may draw patches from, and they
are reprojected to the CRS of each raster. Low-resolution inputs are not needed and
not used. `dataset.py` produces them from every HR patch by area-averaging it down to
`lr_patch_size` and resampling it bicubically back to `patch_size`, which is the
degradation the model learns to invert. The paper trains on 2024 NAIP RGB imagery of
15 coastal California counties within a 10 km coastal buffer, with Santa Barbara
County held out.

Edit `config/config.yaml` to point at your data. It has four sections, `experiment`,
`dataset`, `dataloader` and `trainer`, and the fields that matter most are

| field | meaning |
| --- | --- |
| `experiment.name`, `experiment.log_dir` | name and parent directory of the run directory |
| `dataset.root_dir`, `dataset.masks` | where the imagery and the split masks are |
| `dataset.patch_size` | HR patch, also the tile size used at inference |
| `dataset.lr_patch_size` | the LR grid a patch is degraded to, `round(patch_size / scale)` for a scale factor of `scale` |
| `dataset.stride` | stride of the training patch grid, unrelated to `infer.py --stride` |
| `dataset.bands` | which raster bands to read, 1-indexed |
| `dataset.overlap_threshold`, `dataset.max_nodata_ratio` | how much of a patch must fall inside the mask and how much NoData it may contain |
| `dataset.nodata_value` | the pixel value that counts as NoData, or null to take it from the raster metadata |
| `dataloader.batch_size`, `dataloader.num_workers`, `trainer.max_epochs`, `trainer.accumulate_grad_batches` | the training schedule |

The learning rate, the optimizer and bf16 are not configurable and live in
`sr3/__init__.py` and `train.py`.

Inference needs one raster instead, the low-resolution observation resampled onto the
high-resolution grid. Build it from an HR scene with

```bash
python scripts/create_sr_from_hr.py data/images/scene.tif data/scene.tif --scale 5
```

`--scale` must equal `patch_size / lr_patch_size` of the config you train with,
otherwise the input is degraded differently from the training patches. The second
argument is a name stem rather than a file. The command writes
`data/scene_sr.tif`, the input to `infer.py`, and `data/scene_hr.tif`, the
co-registered reference. Add `--mask some_area.shp` to crop both to the bounds of a
shapefile. If your own input is already a low-resolution raster, upsample it to the
HR grid yourself with `gdalwarp -r cubic` and pass that.

## Training

```bash
python train.py -c config/config.yaml
```

The first run scans every raster, keeps the patches that pass the mask and NoData
tests, and caches that index as a `.pkl` next to the imagery. Scanning a county-sized
raster takes a while, later runs reuse the cache, and changing a patch parameter or
the file list invalidates it. GDAL overviews are built in place on rasters that lack
them, since the NoData test reads them instead of the full resolution.

Checkpoints and TensorBoard logs are written to
`<experiment.log_dir>/<experiment.name>/version_N/`, which is `runs/naip_5x/version_0`
for the configuration as shipped.
Every epoch is kept, and a checkpoint of the model in the paper is 1.9 GB, so budget
the disk for as many epochs as you train. The paper trains for 100 epochs with AdamW
at lr 1e-4, batch 8 per GPU, 8x gradient accumulation and bf16.

Resume an interrupted run from its version directory. The resumed run restores the
weights and the optimizer state from the newest checkpoint in it, and then logs into
the next `version_N`, so the TensorBoard history of one training run is split across
version directories.

```bash
python train.py -c config/config.yaml --resume runs/naip_5x/version_0
```

## Whole-scene inference

```bash
python infer.py -c config/config.yaml \
    --checkpoint runs/naip_5x/version_0/checkpoints/epoch=98-step=91575.ckpt \
    --sr-path data/scene_sr.tif \
    --output outputs/scene \
    --stride 384 --batch-size 8 --amp --pin-memory --seed 42
```

Pass the same config that trained the checkpoint, since the tile size and the bands
come from it. One worker process is started per visible GPU.

| flag | default | meaning |
| --- | --- | --- |
| `--stride` | 384 | tile stride in pixels; 384 with a 512 tile gives the 25% overlap of the paper |
| `--batch-size` | 8 | tiles per forward pass, per GPU |
| `--pin-memory` | off | keep the scene state and the geometry maps in RAM rather than reading them from disk at every step; much faster, at about 60 bytes of host RAM per pixel for a 3-band scene |
| `--amp` | off | run the U-Net under autocast |
| `--num-workers` | 0 | data loader workers, restarted at every timestep; leave at 0 with `--pin-memory` and raise it only when the state is read from disk |
| `--save-freq` | 200 | how often to write the result and a preview, in timesteps |
| `--seed` | unset | seed of the initial scene noise, which fixes the starting point of the reverse chain |
| `--norm-range` | the input's own min and max | scale the input with `lo,hi` instead; pass `0,255` whenever the input is a crop of a larger scene, so that the crop is scaled the way the whole scene would be |
| `--fusion-mode` | `sdvc` | `sdvc` is the method of the paper, `avg` reproduces the naive weighted averaging baseline |

`python infer.py --help` lists the remaining tuning flags.

The output directory receives `image.tif`, the super-resolved scene with the
georeferencing of the input, `preview_<t>.png` at every save point and `preview.png`
at the end, and two working files, `latent.h5` with the scene state and `meta.h5`
with the tiling geometry. Both working files can be deleted once a run is finished.

A run is long. It performs 2000 reverse steps over every tile, and the 4329x4818
scene of the paper takes about four hours on two L4 GPUs. Stop one with Ctrl-C, which
takes a second or two and can leave a warning or two behind as the GPU workers go
down.
Rerunning the identical command continues from the last timestep written to
`latent.h5`, so at most `--save-freq` steps are lost and repeated. Point `--output`
at a fresh directory to start over instead.

## Optional low-frequency projection

The sampler enforces no data consistency, so a whole-scene sample can drift
radiometrically. This post-process re-imposes the observed low frequencies and leaves
the synthesized detail untouched. The paper reports it as a diagnostic, not as part
of the method.

```bash
python lf_project.py --sr outputs/scene/image.tif --obs data/scene_sr.tif \
    --scale 5 --out outputs/scene/image_lfproj.tif
```

## Smoke test

This runs the whole pipeline on a small synthetic scene in a few minutes and needs no
imagery. The model it trains is meaningless, the point is that every stage runs.

```bash
python scripts/make_demo_data.py
python scripts/create_sr_from_hr.py data/demo/images/demo.tif data/demo/scene.tif \
    --scale 5 --mask data/demo/masks/test.shp
CUDA_VISIBLE_DEVICES=0 python train.py -c config/demo.yaml
CUDA_VISIBLE_DEVICES=0 python infer.py -c config/demo.yaml \
    --checkpoint "runs/demo/version_0/checkpoints/epoch=01-step=32.ckpt" \
    --sr-path data/demo/scene_sr.tif --output outputs/demo \
    --stride 96 --batch-size 8 --amp --pin-memory \
    --norm-range 0,255 --save-freq 500 --seed 42
```

Training takes three to four minutes on one L4 and inference about two and a half. The
network is the one from the paper, so the two checkpoints it writes take 3.7 GB. The
result is `outputs/demo/image.tif`, a 256x256 scene fused from nine tiles.

## Citation

```bibtex
@misc{sun2026infscenesr,
  title         = {InfScene-SR: Seamless Super-Resolution of Arbitrarily Large
                   Remote-Sensing Scenes via Variance-Preserving Joint Denoising},
  author        = {Sun, Shoukun and Wang, Zhe and Que, Xiang and Zhang, Jiyin and Ma, Xiaogang},
  year          = {2026},
  eprint        = {2602.19736},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

The diffusion backbone is SR3 (Saharia et al., 2023) and `sr3/unet.py` is adapted
from the reference implementation by Janspiry.
