"""
Observation-guided low-frequency projection (post-process).

Stochastic diffusion SR samplers have no data-consistency term, so whole-scene
samples can drift radiometrically relative to the observation. This script
projects a super-resolved scene back onto the observed low-resolution data:

    SR' = SR + up( down(OBS) - down(SR) )

where down() is the same INTER_AREA operator that defines the task's LR grid,
up() is INTER_CUBIC, and OBS is the *observed input* raster (the bicubically
upsampled LR used as the SR condition) — no high-resolution information is
used. Only frequencies below the LR Nyquist are affected; synthesized
high-frequency content is preserved. This is the digital analogue of the
radiometric normalization step standard in remote-sensing mosaicking.

    python lf_project.py --sr outputs/scene/image.tif \
        --obs data/scene_sr.tif --scale 5 --out outputs/scene/image_lfproj.tif
"""
import argparse

import numpy as np
import cv2
from osgeo import gdal; gdal.UseExceptions()


def read_bands(path, n=3):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()[:n].astype(np.float32)
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    del ds
    return arr, gt, proj


def lf_project(sr, obs, scale):
    c, h, w = sr.shape
    lw, lh = w // scale, h // scale

    def down(img):
        return np.stack([cv2.resize(img[i], (lw, lh),
                                    interpolation=cv2.INTER_AREA)
                         for i in range(c)])

    def up(img):
        return np.stack([cv2.resize(img[i], (w, h),
                                    interpolation=cv2.INTER_CUBIC)
                         for i in range(c)])

    residual = down(obs) - down(sr)
    return np.clip(sr + up(residual), 0, 255)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sr', required=True, help='super-resolved GeoTIFF')
    p.add_argument('--obs', required=True,
                   help='observed input raster (upsampled LR condition)')
    p.add_argument('--scale', type=int, default=5)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    sr, gt, proj = read_bands(args.sr)
    obs, _, _ = read_bands(args.obs)
    h = min(sr.shape[1], obs.shape[1])
    w = min(sr.shape[2], obs.shape[2])
    out = lf_project(sr[:, :h, :w], obs[:, :h, :w], args.scale)
    out = out.round().astype(np.uint8)

    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(args.out, w, h, out.shape[0], gdal.GDT_Byte,
                    options=['COMPRESS=LZW', 'BIGTIFF=YES'])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    for i in range(out.shape[0]):
        ds.GetRasterBand(i + 1).WriteArray(out[i])
    ds.FlushCache()
    del ds
    print(f'wrote {args.out}')
