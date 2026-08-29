"""Build the input raster for whole-scene inference.

The task is defined by its degradation: the HR raster is area-averaged down by
`scale` and bicubically upsampled back to the HR grid. The upsampled result is
what the SR model is conditioned on, and the co-registered HR crop is written
next to it as the reference.
"""
import argparse
import os

import numpy as np
from osgeo import gdal
import cv2
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import CRS

gdal.UseExceptions()

def create_sr_image(input_path, output_path, scale, mask_shapefile=None):
    ds = gdal.Open(input_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open {input_path}")

    full_width = ds.RasterXSize
    full_height = ds.RasterYSize
    bands = ds.RasterCount
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    x_off, y_off = 0, 0
    read_w, read_h = full_width, full_height
    
    if mask_shapefile:
        print(f"Loading mask from {mask_shapefile}")
        gdf = gpd.read_file(mask_shapefile)
        
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * full_width
        miny = maxy + gt[5] * full_height
        
        img_box = box(minx, miny, maxx, maxy)
        
        if gdf.crs is not None and proj:
            try:
                image_crs = CRS.from_wkt(proj)
                if gdf.crs != image_crs:
                     print("Reprojecting mask to image CRS...")
                     gdf = gdf.to_crs(image_crs)
            except Exception as e:
                print(f"Warning: CRS conversion issue: {e}")
        
        valid_geoms = []
        for geom in gdf.geometry:
            if geom.intersects(img_box):
                valid_geoms.append(geom.intersection(img_box))
                
        if not valid_geoms:
            raise ValueError("No intersection between mask shapefile and input image.")
            
        valid_area = unary_union(valid_geoms)
        b_minx, b_miny, b_maxx, b_maxy = valid_area.bounds
        
        # Pixel window of the mask bounds; gt[5] < 0 in north-up data, so the
        # top edge of the window is b_maxy.
        x_off = int((b_minx - gt[0]) / gt[1])
        y_off = int((b_maxy - gt[3]) / gt[5])

        read_w = int(np.ceil((b_maxx - b_minx) / gt[1]))
        read_h = int(np.ceil((b_miny - b_maxy) / gt[5]))
        
        x_off = max(0, x_off)
        y_off = max(0, y_off)
        read_w = min(full_width - x_off, read_w)
        read_h = min(full_height - y_off, read_h)
        
        print(f"Cropping to valid extent: x_off={x_off}, y_off={y_off}, w={read_w}, h={read_h}")

    print("Reading image data...")
    hr = ds.ReadAsArray(x_off, y_off, read_w, read_h)
    
    if bands == 1 and len(hr.shape) == 2:
        hr = hr[np.newaxis, :, :]

    c, h, w = hr.shape
    
    lr_width = w // scale
    lr_height = h // scale
    
    print(f"Processing window from {input_path}")
    print(f"Window size: {w}x{h}, Bands: {c}")
    print(f"Downsampling to: {lr_width}x{lr_height} (Scale: {scale})")

    lr_channels = []
    for i in range(c):
        resized = cv2.resize(
            hr[i],
            (lr_width, lr_height),
            interpolation=cv2.INTER_AREA
        )
        lr_channels.append(resized)
    
    lr = np.stack(lr_channels, axis=0)

    sr_channels = []
    for i in range(c):
        resized = cv2.resize(
            lr[i],
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )
        sr_channels.append(resized)
    
    sr = np.stack(sr_channels, axis=0)

    root, ext = os.path.splitext(output_path)
    hr_output_path = f"{root}_hr{ext}"
    sr_output_path = f"{root}_sr{ext}"

    def save_dataset(path, data):
        driver = gdal.GetDriverByName('GTiff')
        data_type = ds.GetRasterBand(1).DataType
        options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]

        out_ds = driver.Create(path, w, h, c, data_type, options=options)
        
        new_gt = list(gt)
        new_gt[0] = gt[0] + x_off * gt[1] + y_off * gt[2]
        new_gt[3] = gt[3] + x_off * gt[4] + y_off * gt[5]
        
        out_ds.SetGeoTransform(new_gt)
        out_ds.SetProjection(proj)

        for i in range(c):
            out_band = out_ds.GetRasterBand(i + 1)
            out_band.WriteArray(data[i])
            in_band = ds.GetRasterBand(i + 1)
            no_data = in_band.GetNoDataValue()
            if no_data is not None:
                out_band.SetNoDataValue(no_data)
                
        out_ds.FlushCache()
        out_ds = None
        print(f"Saved image to {path}")

    save_dataset(hr_output_path, hr)
    save_dataset(sr_output_path, sr)
    
    ds = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the SR input and its HR reference from an HR raster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_path", help="Input HR GeoTIFF")
    parser.add_argument("output_path",
                        help="Output name stem; '<stem>_sr.tif' and "
                             "'<stem>_hr.tif' are written next to each other")
    parser.add_argument("--scale", type=int, default=5,
                        help="Downsampling factor, which must match "
                             "patch_size / lr_patch_size of the training config")
    parser.add_argument("--mask", default=None,
                        help="Shapefile whose bounds crop both outputs")
    
    args = parser.parse_args()
    
    create_sr_image(args.input_path, args.output_path, args.scale, args.mask)
