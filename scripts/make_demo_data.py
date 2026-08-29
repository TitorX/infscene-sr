"""Write a small synthetic scene and its masks, so the pipeline can be run
end to end without downloading imagery. Not used for any result in the paper.

    python scripts/make_demo_data.py
"""
import argparse
import os

import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from osgeo import gdal, osr; gdal.UseExceptions()

OUT_DIR = "data/demo"
SIZE = 2048  # scene side in pixels
PIXEL = 1.0  # metres per pixel
ORIGIN = (500000.0, 4000000.0)  # top-left corner, EPSG:32610
EPSG = 32610

# Mask boxes in pixel coordinates, (x0, y0, x1, y1).
BOXES = {
    "train": (0, 0, 1024, 2048),
    "val": (1024, 0, 1536, 512),
    "test": (1536, 512, 1792, 768),
}


def fractal_noise(size, rng, octaves=7):
    """Sum of upsampled random grids, which gives detail at every scale."""
    image = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    for octave in range(octaves):
        cells = 2 ** (octave + 2)
        grid = rng.random((cells, cells)).astype(np.float32)
        image += amplitude * cv2.resize(grid, (size, size),
                                        interpolation=cv2.INTER_CUBIC)
        amplitude *= 0.5
    image -= image.min()
    return image / image.max()


def make_scene(rng):
    structure = fractal_noise(SIZE, rng)
    bands = []
    for tint in (1.00, 0.92, 0.78):  # roughly vegetated ground in RGB
        band = 0.75 * structure + 0.25 * fractal_noise(SIZE, rng)
        # 20..235 keeps every pixel away from the NoData value 0.
        bands.append((20 + 215 * tint * band).clip(20, 235).astype(np.uint8))
    return np.stack(bands)


def pixel_box(x0, y0, x1, y1):
    return box(ORIGIN[0] + x0 * PIXEL, ORIGIN[1] - y1 * PIXEL,
               ORIGIN[0] + x1 * PIXEL, ORIGIN[1] - y0 * PIXEL)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/masks", exist_ok=True)

    scene = make_scene(np.random.default_rng(0))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)

    path = f"{OUT_DIR}/images/demo.tif"
    ds = gdal.GetDriverByName("GTiff").Create(
        path, SIZE, SIZE, 3, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform((ORIGIN[0], PIXEL, 0, ORIGIN[1], 0, -PIXEL))
    ds.SetProjection(srs.ExportToWkt())
    for i in range(3):
        ds.GetRasterBand(i + 1).WriteArray(scene[i])
    ds.FlushCache()
    del ds
    print(f"wrote {path} ({SIZE}x{SIZE}, 3 bands)")

    for name, corners in BOXES.items():
        mask_path = f"{OUT_DIR}/masks/{name}.shp"
        gpd.GeoDataFrame(
            {"name": [name]}, geometry=[pixel_box(*corners)], crs=f"EPSG:{EPSG}"
        ).to_file(mask_path)
        print(f"wrote {mask_path}")


if __name__ == "__main__":
    main()
