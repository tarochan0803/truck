# geojson_io.py

import json
import math
import config  # config.py から座標系情報などを参照
from pyproj import Transformer, CRS
from pyproj.exceptions import CRSError
from shapely.geometry import mapping, Point, LineString  # 必要に応じて
import traceback  # エラー出力用

# --- グローバル変数 (Transformer) ---
_transformer_to_meter = None
_transformer_to_latlon = None

# --- 初期化 (Transformer生成) ---
def _initialize_transformers():
    global _transformer_to_meter, _transformer_to_latlon
    if _transformer_to_meter is not None:
        return  # 既に初期化済み

    try:
        source_crs = CRS(f"EPSG:{config.SOURCE_EPSG}")
        target_crs = CRS(f"EPSG:{config.TARGET_EPSG}")
        _transformer_to_meter = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        _transformer_to_latlon = Transformer.from_crs(target_crs, source_crs, always_xy=True)
        print(f"Coordinate Transformers initialized: EPSG:{config.SOURCE_EPSG} <-> EPSG:{config.TARGET_EPSG}")
    except CRSError as e:
        print(f"Error initializing CRS transformers: {e}")
        print(f"Source EPSG: {config.SOURCE_EPSG}, Target EPSG: {config.TARGET_EPSG}")
        # フォールバック: Web Mercator (EPSG:3857) を試みる
        try:
            print("Falling back to Web Mercator (EPSG:3857)...")
            config.TARGET_EPSG = 3857
            target_crs = CRS(f"EPSG:{config.TARGET_EPSG}")
            _transformer_to_meter = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            _transformer_to_latlon = Transformer.from_crs(target_crs, source_crs, always_xy=True)
            print(f"Fallback successful: Using EPSG:{config.TARGET_EPSG}")
        except CRSError as e2:
            print(f"Fallback failed: {e2}")
            _transformer_to_meter = None
            _transformer_to_latlon = None
            raise RuntimeError("Could not initialize coordinate transformers.") from e2


def transform_coords_to_meter(coords_lonlat):
    """
    GeoJSON形式の座標リスト (lon, lat) をメートル座標 (x, y) に変換する。

    引数:
      coords_lonlat: [(lon, lat), ...]
    戻り値:
      [(x, y), ...]
    """
    _initialize_transformers()
    if not _transformer_to_meter:
        raise RuntimeError("Coordinate transformer not initialized.")
    if not coords_lonlat or not all(isinstance(c, (list, tuple)) and len(c) >= 2 for c in coords_lonlat):
        return []
    try:
        coords_m = [_transformer_to_meter.transform(lon, lat) for lon, lat in coords_lonlat]
        return [(x, y) for x, y in coords_m]
    except Exception as e:
        print(f"Error during coordinate transformation to meter: {e}")
        return []


def transform_coords_to_latlon(coords_m):
    """
    メートル座標リスト (x, y) を緯度経度 (lon, lat) に変換する。

    引数:
      coords_m: [(x, y), ...]
    戻り値:
      [(lon, lat), ...]
    """
    _initialize_transformers()
    if not _transformer_to_latlon:
        raise RuntimeError("Coordinate transformer not initialized.")
    if not coords_m or not all(isinstance(c, (list, tuple)) and len(c) >= 2 for c in coords_m):
        return []
    try:
        coords_lonlat = [_transformer_to_latlon.transform(x, y) for x, y in coords_m]
        return [(lon, lat) for lon, lat in coords_lonlat]
    except Exception as e:
        print(f"Error during coordinate transformation to latlon: {e}")
        return []


def calculate_bounds_m(coords_m):
    """
    メートル座標リストから境界 [min_x, min_y, max_x, max_y] を計算する。

    引数:
      coords_m: [(x, y), ...]
    戻り値:
      (min_x, min_y, max_x, max_y) または None
    """
    if not coords_m or not isinstance(coords_m[0], (list, tuple)):
        return None
    try:
        # 1点だけの場合の処理
        if len(coords_m) == 1 and isinstance(coords_m[0][0], (float, int)):
            x, y = coords_m[0]
            return (x, y, x, y)
        min_x = min(c[0] for c in coords_m if isinstance(c, (list, tuple)) and len(c) > 0)
        min_y = min(c[1] for c in coords_m if isinstance(c, (list, tuple)) and len(c) > 1)
        max_x = max(c[0] for c in coords_m if isinstance(c, (list, tuple)) and len(c) > 0)
        max_y = max(c[1] for c in coords_m if isinstance(c, (list, tuple)) and len(c) > 1)
        return (min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f"Error calculating bounds: {e}")
        return None
