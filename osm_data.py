# osm_data.py

import requests
import json
import config  # config.py から座標系情報などを参照
import traceback  # エラー表示用

# geojson_io から必要な関数をインポート
import geojson_io
from geojson_io import transform_coords_to_meter, transform_coords_to_latlon, calculate_bounds_m, _initialize_transformers
from shapely.geometry import mapping, LineString, Point

# Overpass API エンドポイント
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"


def osm_json_to_geojson(osm_data: dict) -> dict:
    """Overpass API から取得した OSM JSON データを GeoJSON FeatureCollection に変換する
       （"out geom;" で提供される geometry を直接利用、建物Polygonに対応）"""
    features = []
    way_count = 0
    relation_count = 0
    processed_feature_count = 0

    print("Starting osm_json_to_geojson processing...")

    for element in osm_data.get("elements", []):
        element_type = element.get("type")

        # --- Way の処理 ---
        if element_type == "way":
            way_count += 1
            way_id = element.get("id")
            tags = element.get("tags", {})
            geometry = element.get("geometry")  # 'out geom;' で得られる座標リスト

            if not way_id or not tags:
                continue  # IDやタグがないWayはスキップ

            # --- 座標リストの抽出 ---
            coordinates = []
            if geometry and isinstance(geometry, list):
                for node_geom in geometry:
                    if (isinstance(node_geom, dict) and 'lat' in node_geom and 'lon' in node_geom and
                        isinstance(node_geom['lon'], (int, float)) and isinstance(node_geom['lat'], (int, float))):
                        coordinates.append((node_geom['lon'], node_geom['lat']))
                    else:
                        # 不正な形式の Node 座標は無視する
                        pass
            else:
                continue  # geometryが存在しない場合はスキップ

            # --- Feature の作成 ---
            properties = tags.copy()
            properties["osm_id"] = way_id
            properties["osm_type"] = "way"
            feature = None

            is_highway = "highway" in tags
            is_building = any(k == "building" or k.startswith("building:") for k in tags)
            # 閉じたWayか判定 (4点以上で始点と終点が同じか)
            is_closed_way = len(coordinates) >= 4 and coordinates[0] == coordinates[-1]

            if is_building and is_closed_way:
                feature = {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": {"type": "Polygon", "coordinates": [coordinates]}
                }
                processed_feature_count += 1
            elif is_highway:
                if len(coordinates) >= 2:
                    feature = {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": {"type": "LineString", "coordinates": coordinates}
                    }
                    processed_feature_count += 1

            if feature:
                features.append(feature)

        # --- Relation の処理 ---
        elif element_type == "relation":
            relation_count += 1
            relation_id = element.get("id")
            tags = element.get("tags", {})
            members = element.get("members")
            # Multipolygon で building タグを持つものを対象 (処理は現状スキップ)
            is_building_multipolygon = (tags.get("type") == "multipolygon" and 
                                        any(k == "building" or k.startswith("building:") for k in tags))
            if is_building_multipolygon and members:
                print(f"Found Building Multipolygon Relation ID: {relation_id} - Processing skipped.")
                # 実装の都合上、関係するWayは別途取得していると仮定する
                pass

        # --- Node の処理 ---
        elif element_type == "node":
            # Node自体は Feature としない
            pass

    print(f"Processed {processed_feature_count} features (LineString/Polygon) from {way_count} ways and {relation_count} relations (Multipolygon skipped).")
    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_osm_highways(bounds_latlon: tuple) -> dict | None:
    """
    指定された範囲 (min_lat, min_lon, max_lat, max_lon) の highway および building データを
    Overpass API から取得し、GeoJSON 形式で返す。
    """
    if len(bounds_latlon) != 4:
        print("Error: Invalid bounds tuple. Expected (min_lat, min_lon, max_lat, max_lon).")
        return None
    if not all(isinstance(v, (int, float)) for v in bounds_latlon):
        print(f"Error: Invalid bounds values: {bounds_latlon}")
        return None
    min_lat, min_lon, max_lat, max_lon = bounds_latlon
    if min_lat > max_lat or min_lon > max_lon:
        print(f"Error: Invalid bounds order: min > max? {bounds_latlon}")
        return None

    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"]({bbox_str});
      (
        way["building"]({bbox_str});
        relation["type"="multipolygon"]["building"]({bbox_str});
      );
    );
    out geom;
    >;
    out skel qt;
    """
    print(f"Fetching OSM data for bbox: {bbox_str}...")
    headers = {'User-Agent': 'TruckSimApp/0.1 (Python requests; contact: your_email@example.com)'}

    try:
        response = requests.post(OVERPASS_API_URL, data={'data': query}, headers=headers, timeout=65)
        response.raise_for_status()

        if response.status_code == 429:
            print("Error 429: Overpass API rate limit exceeded. Please wait.")
            return None
        if response.status_code == 504:
            print("Error 504: Gateway Timeout from Overpass API.")
            return None

        try:
            osm_data_response = response.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding Overpass API JSON response: {e}")
            print(f"Response text (first 500 chars): {response.text[:500]}")
            return None

        if not osm_data_response or "elements" not in osm_data_response:
            print("Warning: No 'elements' found in Overpass API response.")
            osm_data_response = {"elements": []}

        print(f"OSM data received: {len(osm_data_response.get('elements', []))} elements.")
        geojson_data = osm_json_to_geojson(osm_data_response)
        print(f"Converted to GeoJSON: {len(geojson_data.get('features', []))} features found.")
        return geojson_data

    except requests.exceptions.Timeout:
        print("Error: Timeout occurred while fetching data from Overpass API.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Overpass API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}, Text: {e.response.text[:200]}...")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OSM data fetching: {e}")
        traceback.print_exc()
        return None


def get_osm_data_as_meter_geojson(bounds_latlon: tuple) -> tuple[dict | None, tuple | None]:
    """
    指定された範囲の OSM 道路／建物データを取得し、GeoJSON として
    座標をメートル単位に変換した結果と、その境界情報を返す。
    戻り値: (meter_geojson, bounds_m)
    """
    geojson_data_wgs84 = get_osm_highways(bounds_latlon)
    if not geojson_data_wgs84:
        print("Failed to get data from Overpass API or convert to GeoJSON.")
        return None, None
    if not geojson_data_wgs84.get("features"):
        print("No valid features found in OSM data.")
        return {"type": "FeatureCollection", "features": []}, None

    _initialize_transformers()
    if not geojson_io._transformer_to_meter:
        print("ERROR: Coordinate transformer not initialized.")
        raise RuntimeError("Coordinate transformer not initialized.")

    transformed_features = []
    all_coords_m = []
    processed_count = 0
    feature_count = len(geojson_data_wgs84["features"])

    for feature in geojson_data_wgs84["features"]:
        geom = feature.get("geometry")
        if not geom:
            continue
        geom_type = geom.get("type")
        coords_wgs84 = geom.get("coordinates")
        if geom_type not in ["LineString", "Polygon"] or not coords_wgs84:
            continue

        new_feature = feature.copy()
        new_geom = geom.copy()
        feature_coords_m = []

        try:
            if geom_type == "LineString":
                coords_m = transform_coords_to_meter(coords_wgs84)
                if len(coords_m) < 2:
                    continue
                new_geom["coordinates"] = coords_m
                feature_coords_m = coords_m
            elif geom_type == "Polygon":
                transformed_rings = []
                valid_polygon = True
                for i, ring_wgs84 in enumerate(coords_wgs84):
                    coords_m = transform_coords_to_meter(ring_wgs84)
                    if len(coords_m) < 3:
                        print(f"Warn: Ring {i} for Polygon {feature.get('properties', {}).get('osm_id','N/A')} has < 3 points after transform.")
                        valid_polygon = False
                        break
                    transformed_rings.append(coords_m)
                    if i == 0:
                        feature_coords_m.extend(coords_m)
                if not valid_polygon:
                    continue
                new_geom["coordinates"] = transformed_rings
            new_feature["geometry"] = new_geom
            transformed_features.append(new_feature)
            all_coords_m.extend(feature_coords_m)
            processed_count += 1
        except Exception as e:
            osm_id = feature.get("properties", {}).get("osm_id", "N/A")
            print(f"Error transforming feature {osm_id} (type: {geom_type}): {e}")

    if feature_count > 0:
        print(f"OSM data processed: {processed_count}/{feature_count} features transformed to meters.")
    else:
        print("No features found in the initial OSM data.")

    meter_geojson = {
        "type": "FeatureCollection",
        "features": transformed_features,
        "crs": {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{config.TARGET_EPSG}"
            }
        }
    }
    bounds_m = calculate_bounds_m(all_coords_m) if all_coords_m else None
    return meter_geojson, bounds_m


def export_path_to_geojson(path_coords_m: list[dict], filepath: str):
    """パス (メートル座標リスト [{'x': m, 'y': m}]) を LineString GeoJSON で保存する"""
    if not path_coords_m or len(path_coords_m) < 2:
        raise ValueError("Path must contain at least 2 points.")
    coords_for_export = [(p['x'], p['y']) for p in path_coords_m]
    feature = _create_geojson_feature("LineString", coords_for_export, {"name": "Simulation Path"})
    if not feature:
        raise RuntimeError("Failed to create GeoJSON feature for the path.")
    geojson_output = {
        "type": "FeatureCollection",
        "features": [feature],
        "crs": {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{config.TARGET_EPSG}"
            }
        }
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson_output, f, indent=2)
        print(f"Path exported successfully to {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error writing GeoJSON file {filepath}: {e}")


def export_simulation_results_to_geojson(truck_history_m: list[dict], corner_history_m: list[list[dict]], filepath: str):
    """シミュレーション結果 (軌跡) を GeoJSON で保存する"""
    features = []

    if truck_history_m and len(truck_history_m) >= 2:
        center_coords = [(h['x_m'], h['y_m']) for h in truck_history_m]
        center_feature = _create_geojson_feature("LineString", center_coords, {"name": "Truck Center Trajectory"})
        if center_feature:
            features.append(center_feature)

    if corner_history_m and len(corner_history_m) >= 2:
        try:
            num_corners = len(corner_history_m[0])
            for i in range(num_corners):
                corner_coords = []
                valid_corner_traj = True
                for step_index in range(len(corner_history_m)):
                    if i < len(corner_history_m[step_index]):
                        corner_m = corner_history_m[step_index][i]
                        if isinstance(corner_m, dict) and 'x' in corner_m and 'y' in corner_m:
                            corner_coords.append((corner_m['x'], corner_m['y']))
                        else:
                            valid_corner_traj = False
                            break
                    else:
                        valid_corner_traj = False
                        break
                if valid_corner_traj and len(corner_coords) >= 2:
                    corner_feature = _create_geojson_feature("LineString", corner_coords, {"name": f"Corner {i+1} Trajectory"})
                    if corner_feature:
                        features.append(corner_feature)
        except Exception as e:
            print(f"Error processing corner trajectory {i+1}: {e}")

    if not features:
        raise ValueError("No valid simulation data to export.")

    geojson_output = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{config.TARGET_EPSG}"
            }
        }
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson_output, f, indent=2)
        print(f"Simulation results exported successfully to {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error writing GeoJSON file {filepath}: {e}")


def _create_geojson_feature(geometry_type, coords_m, properties=None) -> dict | None:
    """メートル座標リストから GeoJSON Feature を作成する (最初に緯度経度に変換)"""
    if not coords_m:
        return None
    try:
        coords_lonlat = transform_coords_to_latlon(coords_m)
        if not coords_lonlat:
            print("Warning: Failed to transform coordinates back to lat/lon for export.")
            return None

        if geometry_type == "LineString":
            if len(coords_lonlat) < 2:
                return None
            geom = LineString(coords_lonlat)
        elif geometry_type == "Point":
            if len(coords_lonlat) != 1:
                return None
            geom = Point(coords_lonlat[0])
        else:
            print(f"Unsupported geometry type for export: {geometry_type}")
            return None

        return {
            "type": "Feature",
            "properties": properties if properties else {},
            "geometry": mapping(geom)
        }
    except Exception as e:
        print(f"Error creating GeoJSON feature: {e}")
        return None
