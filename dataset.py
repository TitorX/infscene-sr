import os
import glob
import pickle
import hashlib
from typing import List, Tuple, Dict, Optional
from collections import namedtuple
import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import Dataset
import cv2
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import geopandas as gpd
import warnings
from tqdm import tqdm
from pyproj import CRS


gdal.UseExceptions()

PatchMetadata = namedtuple(
    "PatchMetadata",
    [
        "filename",
        "x_offset",
        "y_offset",
        "width",
        "height",
        "geotransform",
        "projection",
        "overlap_ratio",
    ],
)


class RSSRDataset(Dataset):
    """
    Remote sensing super-resolution (RSSR) dataset for remote sensing images.

    Args:
        root_dir: Root directory containing image files (searches recursively)
        mask_shapefile: Path to shapefile indicating valid areas
        patch_size: Size of HR patches (e.g., 256)
        stride: Stride for patch grid generation
        lr_patch_size: Size of LR patches (e.g., 64)
        bands: List of band indices to read (1-indexed), e.g., [1, 2, 3] or [3, 1, 2]
        overlap_threshold: Minimum overlap ratio to keep a patch (default: 0.8)
        max_nodata_ratio: Maximum allowed ratio of NoData pixels in a patch (default: 0.2)
        nodata_value: Optional NoData value for filtering

    Note:
        Supported image formats: .tif, .tiff, .sid (MrSID)
        The dataset will recursively search all subdirectories under root_dir
    """

    def __init__(
        self,
        root_dir: str,
        mask_shapefile: str,
        patch_size: int = 256,
        stride: int = 128,
        lr_patch_size: int = 64,
        bands: List[int] = [1, 2, 3, 4],
        overlap_threshold: float = 0.8,
        max_nodata_ratio: float = 0.2,
        nodata_value: Optional[float] = None,
    ):
        self.root_dir = root_dir
        self.mask_shapefile = mask_shapefile
        self.patch_size = patch_size
        self.stride = stride
        self.lr_patch_size = lr_patch_size
        self.bands = bands
        self.overlap_threshold = overlap_threshold
        self.max_nodata_ratio = max_nodata_ratio
        self.nodata_value = nodata_value

        self.tif_files = self._find_tif_files()

        self.mask_gdf = gpd.read_file(mask_shapefile)

        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            print(f"Loading patches from cache: {cache_path}")
            self.patches = self._load_cache(cache_path)
            print(
                f"Dataset initialized with {len(self.patches)} valid patches (from cache)"
            )
        else:
            self.patches = self._generate_patches()
            print(f"Dataset initialized with {len(self.patches)} valid patches")

            self._save_cache(cache_path)
            print(f"Saved patches to cache: {cache_path}")

    def _get_cache_path(self) -> str:
        """Cache path keyed by the mask name, the patch parameters and the file
        list, so that changing any of them invalidates the cached patch index.
        Editing a mask in place without renaming it does not."""
        mask_basename = os.path.splitext(os.path.basename(self.mask_shapefile))[0]

        cache_key = f"{mask_basename}_ps{self.patch_size}_s{self.stride}_lr{self.lr_patch_size}_ot{self.overlap_threshold}_mn{self.max_nodata_ratio}"

        tif_files_str = "|".join(sorted(self.tif_files))
        tif_hash = hashlib.md5(tif_files_str.encode()).hexdigest()[:8]
        cache_key = f"{cache_key}_{tif_hash}"

        cache_filename = f"patches_cache_{cache_key}.pkl"
        return os.path.join(self.root_dir, cache_filename)

    def _save_cache(self, cache_path: str) -> None:
        """Save patches to cache file."""
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.patches, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}", UserWarning)

    def _load_cache(self, cache_path: str) -> List[Dict]:
        """Load patches from cache file."""
        try:
            with open(cache_path, "rb") as f:
                patches = pickle.load(f)
            return patches
        except Exception as e:
            warnings.warn(
                f"Failed to load cache: {e}. Regenerating patches...", UserWarning
            )
            return []

    @staticmethod
    def _get_nodata_value(
        filename: str, band_idx: int, fallback: Optional[float] = None
    ) -> Optional[float]:
        """Get nodata value from file or use fallback."""
        if fallback is not None:
            return fallback

        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            return None

        band = ds.GetRasterBand(band_idx)
        nodata_val = band.GetNoDataValue()
        ds = None
        return nodata_val

    @staticmethod
    def _calculate_overview_scale(full_size: int, overview_size: int) -> float:
        """Calculate scale factor between full resolution and overview."""
        return full_size / overview_size

    def _ensure_overview(self, tif_file: str, overview_level: int = 16) -> bool:
        """Ensure that overviews exist for the image file. Create if needed.

        Args:
            tif_file: Path to the image file
            overview_level: Overview level (e.g., 16 for 1/16 scale)

        Returns:
            True if overviews exist or were created successfully
        """
        ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
        if ds is None:
            return False

        band = ds.GetRasterBand(1)

        overview_count = band.GetOverviewCount()
        has_overview = False

        for i in range(overview_count):
            overview = band.GetOverview(i)
            avg_scale = (
                self._calculate_overview_scale(ds.RasterXSize, overview.XSize)
                + self._calculate_overview_scale(ds.RasterYSize, overview.YSize)
            ) / 2

            if abs(avg_scale - overview_level) / overview_level < 0.5:
                has_overview = True
                break

        ds = None

        if not has_overview:
            print(f"  Creating overviews for {os.path.basename(tif_file)}...")
            ds = gdal.Open(tif_file, gdal.GA_Update)
            if ds is None:
                ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
                if ds is None:
                    return False

            # AVERAGE resampling makes partially-nodata blocks visible.
            result = ds.BuildOverviews("AVERAGE", [8, 16, 32])
            ds = None

            if result != 0:
                warnings.warn(f"Failed to create overviews for {tif_file}", UserWarning)
                return False

            print(f"  Overviews created successfully")

        return True

    def _load_overview(
        self, tif_file: str, band_idx: int = 1, target_level: int = 16
    ) -> Optional[np.ndarray]:
        """Load a full overview into memory for fast access.

        Args:
            tif_file: Path to the image file
            band_idx: Band index to read (1-indexed)
            target_level: Target overview level (e.g., 16 for 1/16 scale)

        Returns:
            Overview data as numpy array, or None if not available
        """
        ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
        if ds is None:
            return None

        band = ds.GetRasterBand(band_idx)
        overview_count = band.GetOverviewCount()

        best_overview = None
        best_scale_diff = float("inf")

        for i in range(overview_count):
            overview = band.GetOverview(i)
            avg_scale = (
                self._calculate_overview_scale(ds.RasterXSize, overview.XSize)
                + self._calculate_overview_scale(ds.RasterYSize, overview.YSize)
            ) / 2

            scale_diff = abs(avg_scale - target_level)
            if scale_diff < best_scale_diff:
                best_scale_diff = scale_diff
                best_overview = overview

        if best_overview is None:
            ds = None
            return None

        overview_data = best_overview.ReadAsArray()
        overview_scale = self._calculate_overview_scale(
            ds.RasterXSize, best_overview.XSize
        )

        ds = None

        return overview_data, overview_scale

    def _find_tif_files(self) -> List[str]:
        """Find all .tif and .sid files recursively in the root directory."""
        tif_files = []
        extensions = ["*.tif", "*.tiff", "*.sid"]

        for ext in extensions:
            tif_files.extend(
                glob.glob(os.path.join(self.root_dir, "**", ext), recursive=True)
            )

        if not tif_files:
            raise ValueError(
                f"No .tif/.tiff/.sid files found in {self.root_dir} (searched recursively)"
            )

        print(f"Found {len(tif_files)} image files (searching recursively)")
        return sorted(tif_files)

    def _get_image_extent(self, tif_file: str) -> Tuple[Polygon, Dict]:
        """Get the geographic extent of a tif file as a polygon."""
        ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Cannot open file: {tif_file}")

        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        width = ds.RasterXSize
        height = ds.RasterYSize

        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + geotransform[1] * width
        miny = maxy + geotransform[5] * height

        ds = None

        extent_polygon = box(minx, miny, maxx, maxy)

        return extent_polygon, {
            "geotransform": geotransform,
            "projection": projection,
            "width": width,
            "height": height,
        }

    def _calculate_valid_area(self, tif_file: str) -> Polygon:
        """Calculate valid area for a tif file based on mask shapefile."""
        extent_polygon, image_info = self._get_image_extent(tif_file)

        mask_gdf = self.mask_gdf.copy()
        if mask_gdf.crs is not None:
            try:
                image_crs = CRS.from_wkt(image_info["projection"])
                mask_gdf = mask_gdf.to_crs(image_crs)
            except Exception as e:
                warnings.warn(
                    f"Failed to convert CRS for mask shapefile. Error: {e}. Assuming same CRS as image: {tif_file}",
                    UserWarning,
                )

        valid_geoms = []
        for geom in mask_gdf.geometry:
            if geom.intersects(extent_polygon):
                intersection = geom.intersection(extent_polygon)
                if not intersection.is_empty:
                    valid_geoms.append(intersection)

        if not valid_geoms:
            return Polygon()  # Empty polygon

        valid_area = unary_union(valid_geoms)

        return valid_area

    def _pixel_to_geo(
        self, px: int, py: int, geotransform: Tuple
    ) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates."""
        geo_x = geotransform[0] + px * geotransform[1] + py * geotransform[2]
        geo_y = geotransform[3] + px * geotransform[4] + py * geotransform[5]
        return geo_x, geo_y

    def _generate_patch_grid(
        self,
        tif_file: str,
        image_info: Dict,
        valid_area: Polygon,
        overview_data: Optional[np.ndarray] = None,
        overview_scale: float = 1.0,
    ) -> List[Dict]:
        """Generate patch grid for a tif file.

        Args:
            tif_file: Path to the image file
            image_info: Dictionary with image metadata
            valid_area: Polygon defining valid areas
            overview_data: Cached overview data for nodata checking
            overview_scale: Scale factor of the overview (e.g., 16.0 for 1/16)
        """
        patches = []
        geotransform = image_info["geotransform"]
        width = image_info["width"]
        height = image_info["height"]

        y_range = list(range(0, height - self.patch_size + 1, self.stride))
        x_range = list(range(0, width - self.patch_size + 1, self.stride))
        total_tiles = len(y_range) * len(x_range)

        with tqdm(total=total_tiles, desc="  Generating patches") as pbar:
            for y in y_range:
                for x in x_range:
                    x1_geo, y1_geo = self._pixel_to_geo(x, y, geotransform)
                    x2_geo, y2_geo = self._pixel_to_geo(
                        x + self.patch_size, y + self.patch_size, geotransform
                    )

                    minx = min(x1_geo, x2_geo)
                    maxx = max(x1_geo, x2_geo)
                    miny = min(y1_geo, y2_geo)
                    maxy = max(y1_geo, y2_geo)

                    patch_box = box(minx, miny, maxx, maxy)

                    if valid_area.is_empty:
                        pbar.update(1)
                        continue

                    intersection = patch_box.intersection(valid_area)
                    overlap_ratio = intersection.area / patch_box.area

                    if overlap_ratio >= self.overlap_threshold:
                        patch_geotransform = list(geotransform)
                        patch_geotransform[0] = x1_geo  # top-left x
                        patch_geotransform[3] = y1_geo  # top-left y

                        patch_candidate = {
                            "filename": tif_file,
                            "x_offset": x,
                            "y_offset": y,
                            "width": self.patch_size,
                            "height": self.patch_size,
                            "geotransform": tuple(patch_geotransform),
                            "projection": image_info["projection"],
                            "overlap_ratio": overlap_ratio,
                        }

                        nodata_ratio = self._check_nodata_ratio(
                            patch_candidate, overview_data, overview_scale
                        )

                        if nodata_ratio <= self.max_nodata_ratio:
                            patches.append(patch_candidate)

                    pbar.update(1)

        return patches

    def _generate_patches(self) -> List[Dict]:
        """Generate all valid patches from all tif files."""
        all_patches = []

        for tif_file in self.tif_files:
            extent_polygon, image_info = self._get_image_extent(tif_file)

            width = image_info["width"]
            height = image_info["height"]
            print(f"Processing {os.path.basename(tif_file)} ({width}x{height})...")

            valid_area = self._calculate_valid_area(tif_file)

            if valid_area.is_empty:
                print(f"  No valid area found, skipping...")
                print("=" * 50)
                continue

            overview_data = None
            overview_scale = 1.0
            if self._ensure_overview(tif_file, overview_level=16):
                result = self._load_overview(
                    tif_file, band_idx=self.bands[0], target_level=16
                )
                if result is not None:
                    overview_data, overview_scale = result
                    print(
                        f"  Loaded 1/{int(overview_scale)} overview into memory for fast nodata checking"
                    )
                else:
                    print(
                        f"  Warning: Could not load overview, will use full resolution (slower)"
                    )
            else:
                print(
                    f"  Warning: Could not create overviews, will use full resolution (slower)"
                )

            patches = self._generate_patch_grid(
                tif_file, image_info, valid_area, overview_data, overview_scale
            )
            all_patches.extend(patches)

            print(f"  Generated {len(patches)} valid patches")
            print("=" * 50)

        return all_patches

    def _check_nodata_ratio(
        self,
        patch_info: Dict,
        overview_data: Optional[np.ndarray] = None,
        overview_scale: float = 1.0,
    ) -> float:
        """Calculate the ratio of nodata pixels in a patch.

        Args:
            patch_info: Patch metadata dictionary
            overview_data: Cached overview data for fast access
            overview_scale: Scale factor of the overview

        Returns:
            Ratio of nodata pixels (0.0 to 1.0)
        """
        nodata_val = self._get_nodata_value(
            patch_info["filename"], self.bands[0], self.nodata_value
        )
        if nodata_val is None:
            return 0.0

        # An overview is 16x coarser, which turns a full-resolution read of
        # every candidate patch into a slice of an array already in memory.
        if overview_data is not None:
            overview_x = int(patch_info["x_offset"] / overview_scale)
            overview_y = int(patch_info["y_offset"] / overview_scale)
            overview_w = max(1, int(patch_info["width"] / overview_scale))
            overview_h = max(1, int(patch_info["height"] / overview_scale))

            overview_h_max, overview_w_max = overview_data.shape
            overview_x = min(overview_x, overview_w_max - 1)
            overview_y = min(overview_y, overview_h_max - 1)
            overview_w = min(overview_w, overview_w_max - overview_x)
            overview_h = min(overview_h, overview_h_max - overview_y)

            data = overview_data[
                overview_y : overview_y + overview_h,
                overview_x : overview_x + overview_w,
            ]
        else:
            ds = gdal.Open(patch_info["filename"], gdal.GA_ReadOnly)
            if ds is None:
                return 0.0

            band = ds.GetRasterBand(self.bands[0])
            data = band.ReadAsArray(
                patch_info["x_offset"],
                patch_info["y_offset"],
                patch_info["width"],
                patch_info["height"],
            )
            ds = None

        if data is None or data.size == 0:
            return 0.0

        total_pixels = data.size
        nodata_pixels = np.sum(np.isclose(data, nodata_val, rtol=1e-5))
        nodata_ratio = nodata_pixels / total_pixels

        return nodata_ratio

    def _read_patch(self, patch_info: Dict) -> np.ndarray:
        """Read a patch from a tif file using GDAL."""
        ds = gdal.Open(patch_info["filename"], gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Cannot open file: {patch_info['filename']}")

        patch_data = []
        for band_idx in self.bands:
            band = ds.GetRasterBand(band_idx)
            data = band.ReadAsArray(
                patch_info["x_offset"],
                patch_info["y_offset"],
                patch_info["width"],
                patch_info["height"],
            )

            nodata_val = self._get_nodata_value(
                patch_info["filename"], band_idx, self.nodata_value
            )
            if nodata_val is not None:
                data[np.isclose(data, nodata_val, rtol=1e-5)] = 0

            patch_data.append(data)

        ds = None

        return np.stack(patch_data, axis=0)

    def _create_lr(self, hr: np.ndarray) -> np.ndarray:
        """Create low-resolution image from high-resolution."""
        c, h, w = hr.shape

        lr = np.stack(
            [
                cv2.resize(
                    hr[i],
                    (self.lr_patch_size, self.lr_patch_size),
                    interpolation=cv2.INTER_AREA,
                )
                for i in range(c)
            ],
            axis=0,
        )

        return np.stack(
            [
                cv2.resize(lr[i], (w, h), interpolation=cv2.INTER_CUBIC)
                for i in range(c)
            ],
            axis=0,
        )

    def __len__(self) -> int:
        """Return the number of valid patches."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Get a sample from the dataset.

        Returns:
            hr: High-resolution image (C, H, W) as tensor in [-1, 1]
            lr: Low-resolution image upsampled to HR size (C, H, W) as tensor in [-1, 1]
            meta: Dictionary containing patch metadata
        """
        patch_info = self.patches[idx]

        hr = self._read_patch(patch_info)
        lr = self._create_lr(hr)

        meta = PatchMetadata(
            filename=os.path.basename(patch_info["filename"]),
            x_offset=patch_info["x_offset"],
            y_offset=patch_info["y_offset"],
            width=patch_info["width"],
            height=patch_info["height"],
            geotransform=patch_info["geotransform"],
            projection=patch_info["projection"],
            overlap_ratio=patch_info["overlap_ratio"],
        )

        # uint8 [0, 255] -> float32 [-1, 1]
        hr = torch.from_numpy(hr.astype(np.float32) / 255.0) * 2.0 - 1.0
        lr = torch.from_numpy(lr.astype(np.float32) / 255.0) * 2.0 - 1.0

        return hr, lr, meta
