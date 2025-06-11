#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
【概要】
・GeoJSONデータに基づく道路情報と、パス上のウェイポイントをもとに
  トラックの走行挙動（PurePursuit + バイシクルモデル + PID + SMC 制御）をシミュレーションし、
  曲率誤差などを評価するコードです。
・シミュレーション結果は、CSV出力や最適化、またはGUI上での描画に利用できます。
・今回の改良点として、報酬関数を以下の要素で設計しています。
  1. クロストラック誤差のペナルティ  
  2. 進行方向（ヘディング）のずれのペナルティ  
  4. 制御入力のペナルティ  
  5. 目標到達ボーナス
  ※速度誤差は今回は採用していません。
  
【注意】
・重み（α, β, δ, ボーナス値）は例として設定しており、実際の用途に合わせてチューニングしてください。
"""

import math
import json
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from datetime import datetime

import numpy as np
from shapely.geometry import LineString, Point
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time  # 停止判定用
import csv   # CSV出力用

# --------------------------------------------------
# 1. 曲率計算用のヘルパー関数
# --------------------------------------------------
def curvature_of_three_points(x0, y0, x1, y1, x2, y2, deg_per_m):
    """
    3点の座標から曲率を計算する。
    度単位の座標を、メートル単位に変換して計算します。
    """
    x0_m = x0 / deg_per_m
    y0_m = y0 / deg_per_m
    x1_m = x1 / deg_per_m
    y1_m = y1 / deg_per_m
    x2_m = x2 / deg_per_m
    y2_m = y2 / deg_per_m

    a = math.dist((x0_m, y0_m), (x1_m, y1_m))
    b = math.dist((x1_m, y1_m), (x2_m, y2_m))
    c = math.dist((x0_m, y0_m), (x2_m, y2_m))

    area = abs((x1_m - x0_m) * (y2_m - y0_m) - (x2_m - x0_m) * (y1_m - y0_m)) / 2.0
    if area == 0:
        return 0.0
    curvature = (4.0 * area) / (a * b * c)
    return curvature


def curvature_along_polyline(points, deg_per_m):
    """
    ポリライン上の各点（中間点）の曲率を計算する。
    端点は0とする（簡易実装）。
    """
    n = len(points)
    if n < 3:
        return [0.0] * n
    kappas = [0.0] * n
    for i in range(1, n - 1):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        kappas[i] = curvature_of_three_points(x0, y0, x1, y1, x2, y2, deg_per_m)
    return kappas


def curvature_cost(path_points, corner_points, deg_per_m):
    """
    2つのポリライン（参考パスと車両四隅の軌跡）の曲率誤差（二乗和）を計算する。
    """
    k1 = curvature_along_polyline(path_points, deg_per_m)
    k2 = curvature_along_polyline(corner_points, deg_per_m)
    n = min(len(k1), len(k2))
    if n == 0:
        return 0.0
    err_sum = 0.0
    for i in range(n):
        diff = k1[i] - k2[i]
        err_sum += diff * diff
    return err_sum


def overall_curvature_mismatch(path_points, corner_trajs, deg_per_m):
    """
    全4隅の曲率誤差を合算して返す。
    corner_trajs: [corner0_points, corner1_points, corner2_points, corner3_points]
    """
    total = 0.0
    for cpts in corner_trajs:
        total += curvature_cost(path_points, cpts, deg_per_m)
    return total


# --------------------------------------------------
# 2. 角度系のユーティリティ関数
# --------------------------------------------------
def deg_to_rad(deg):
    """度をラジアンに変換する。"""
    return deg * math.pi / 180.0


def normalize_angle(a):
    """
    角度 a (ラジアン) を -pi～pi に正規化する。
    """
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# --------------------------------------------------
# 3. GeoJsonTruckCanvas: シミュレーション＆描画クラス
# --------------------------------------------------
class GeoJsonTruckCanvas(tk.Canvas):
    """
    PurePursuit + バイシクルモデル + PID + SMC 制御などを組み合わせた
    トラックシミュレーションおよび描画用の Canvas クラス。

    ※60秒以上 sim_step が呼ばれないと停止し、MainApp 側で再実行されます。
    """
    def __init__(self, parent, master_app=None, width=800, height=600, bg="white"):
        super().__init__(parent, width=width, height=height, bg=bg)
        self.pack(fill=tk.BOTH, expand=True)
        
        self.master_app = master_app
        self.geojson_data = None
        self.bounds = None
        self.geo_points = []
        self.selected_road_featID = None
        self.meter_per_deg = 111320.0  # 1度あたりの平均距離（メートル）

        self.margin = 20
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.rotation = 0.0
        self.line_item_to_feature = {}

        # 中ボタンドラッグ用
        self.drag_start_x = 0
        self.drag_start_y = 0

        # 車両パラメータ
        self.wheelBase_m = 4.0
        self.frontOverhang_m = 1.0
        self.rearOverhang_m = 1.0
        self.vehicleWidth_m = 2.5
        self.maxSteeringAngle_deg = 45
        self.vehicleSpeed_m_s = 5.0
        self.lookahead_m = 15.0

        # バイシクルモデル用（度単位から変換後）
        self.wheelBase_deg = 0
        self.frontOverhang_deg = 0
        self.rearOverhang_deg = 0
        self.vehicleWidth_deg = 0
        self.maxSteeringAngle = 0
        self.vehicleSpeed_deg_s = 0
        self.lookahead_deg = 0

        # 物理モデル
        self.max_steer_rate = deg_to_rad(30)
        self.vehicle_mass = 5000
        self.max_accel = 0.3
        self.max_brake = 3.0
        self.drag_coeff = 0.35
        self.roll_resist = 0.015

        # PID / SMC 用
        self.pid_kp = 1.5
        self.pid_ki = 0.2
        self.pid_kd = 0.1
        self.error_integral = 0.0
        self.last_error = 0.0

        self.smc_lambda = 0.8
        self.smc_eta = 0.2

        # 速度PID
        self.speed_kp = 1.0
        self.speed_ki = 0.1
        self.speed_kd = 0.05
        self.speed_integral = 0.0
        self.last_speed_error = 0.0

        # シミュレーション関連
        self.path = []          # ユーザーが追加するウェイポイント
        self.running = False    # シミュレーション実行フラグ
        self.animation_id = None
        self.dt = 0.1

        # 状態変数（車両の現在位置・姿勢・速度・舵角）
        self.truck_x = 0
        self.truck_y = 0
        self.truck_theta = 0
        self.truck_velocity = 0
        self.truck_steering = 0

        # ログ関連
        self.corner_trajs = [[], [], [], []]  # 車両四隅の軌跡
        self.history = []                     # シミュレーション履歴
        self.alert_marker = None

        # KDTree（パス点探索用）
        self.path_tree = None

        # コースアウト判定（今回使わないが保持）
        self.corners_outside = None

        # イベントバインド
        self.bind("<Button-1>", self.on_left_click)
        self.bind("<ButtonPress-2>", self.on_mid_down)
        self.bind("<B2-Motion>", self.on_mid_drag)
        self.bind("<ButtonRelease-2>", self.on_mid_up)
        self.bind("<MouseWheel>", self.on_mousewheel)

    # 変換ユーティリティ
    def deg_to_m(self, deg):
        return deg * self.meter_per_deg
        
    def m_to_deg(self, m):
        return m / self.meter_per_deg

    # KDTree の構築
    def build_path_tree(self):
        """現在の self.path から KDTree を構築する。"""
        if self.path:
            pts = [(pt["x"], pt["y"]) for pt in self.path]
            self.path_tree = KDTree(pts)
        else:
            self.path_tree = None

    # ------------- GeoJSON 読み込み関連 -------------
    def load_geojson(self, fp):
        with open(fp, "r", encoding="utf-8") as f:
            self.geojson_data = json.load(f)
        self.extract_geo_points()
        self.compute_bounds()
        self.full_view()
        self.redraw()
        self.path_tree = None

    def extract_geo_points(self):
        """GeoJSON内の全座標を抽出して self.geo_points に格納する。"""
        self.geo_points = []
        if not self.geojson_data:
            return
        feats = self.geojson_data.get("features", [])
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                for c_ in coords:
                    self.geo_points.append((c_[0], c_[1]))
            elif geom.get("type") == "Point":
                c_ = geom.get("coordinates", [])
                self.geo_points.append((c_[0], c_[1]))

    def compute_bounds(self):
        """GeoJSON の全座標から表示範囲（bounds）を計算する。"""
        if not self.geojson_data:
            self.bounds = (0, 1, 0, 1)
            return
        lons, lats = [], []
        feats = self.geojson_data.get("features", [])
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                for c_ in geom.get("coordinates", []):
                    lons.append(c_[0])
                    lats.append(c_[1])
            elif geom.get("type") == "Point":
                c_ = geom.get("coordinates", [])
                lons.append(c_[0])
                lats.append(c_[1])
        if lons and lats:
            self.bounds = (min(lons), max(lons), min(lats), max(lats))
        else:
            self.bounds = (0, 1, 0, 1)

    def full_view(self):
        """キャンバス全体に合わせた表示設定を行う。"""
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return
        (minx, maxx, miny, maxy) = self.bounds
        dx = maxx - minx
        dy = maxy - miny
        if dx < 1e-9 or dy < 1e-9:
            return
        vw = w - 2 * self.margin
        vh = h - 2 * self.margin
        scx = vw / dx
        scy = vh / dy
        s = min(scx, scy)
        cx, cy = w * 0.5, h * 0.5
        midx = (minx + maxx) * 0.5
        midy = (miny + maxy) * 0.5
        self.scale_factor = s
        self.offset_x = cx - s * midx
        self.offset_y = cy + s * midy

    def deg_per_m(self):
        """
        1mあたりの度数（地理座標系の場合）を計算する。
        """
        if not self.bounds:
            return 1e-6
        (minx, maxx, miny, maxy) = self.bounds
        lat_c = (miny + maxy) * 0.5
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(lat_c))
        meter_per_deg = (lat_m + lon_m) * 0.5
        return 1.0 / meter_per_deg

    # ------------- 描画関連 -------------
    def redraw(self):
        """キャンバス上の全要素を再描画する。"""
        self.delete("all")
        # 道路描画
        if self.geojson_data:
            feats = self.geojson_data.get("features", [])
            for i, ft in enumerate(feats):
                geom = ft.get("geometry", {})
                props = ft.get("properties", {})
                road_m = props.get("roadWidthM", 2.0)
                color_ = props.get("color", "gray")
                if i == self.selected_road_featID:
                    color_ = "orange"
                road_deg = road_m * self.deg_per_m()
                road_px = self.deg_to_px(road_deg)
                if geom.get("type") == "LineString":
                    coords = geom.get("coordinates", [])
                    if len(coords) > 1:
                        arr = []
                        for c_ in coords:
                            px, py = self.model_to_canvas(c_[0], c_[1])
                            arr.extend([px, py])
                        self.create_line(*arr, fill=color_, width=road_px)

        # パス描画
        if len(self.path) > 1:
            arr = []
            for p_ in self.path:
                px, py = self.model_to_canvas(p_['x'], p_['y'])
                arr.extend([px, py])
            self.create_line(*arr, fill="blue", width=2)

        # 角の軌跡描画
        colz = ["red", "green", "blue", "orange"]
        for i, traj in enumerate(self.corner_trajs):
            if len(traj) < 2:
                continue
            arr = []
            for wpt in traj:
                sx, sy = self.model_to_canvas(wpt["x"], wpt["y"])
                arr.extend([sx, sy])
            self.create_line(*arr, fill=colz[i], dash=(2, 2), width=1)

        # トラック描画
        if self.history:
            self.draw_truck()

    def deg_to_px(self, dval):
        """モデル座標の長さをピクセル単位に変換する。"""
        x1, y1 = self.model_to_canvas(0, 0)
        x2, y2 = self.model_to_canvas(dval, 0)
        return max(1, int(math.hypot(x2 - x1, y2 - y1)))

    def model_to_canvas(self, x, y):
        """
        モデル座標 (x, y) をキャンバス座標に変換する。
        回転・スケール・オフセットを考慮する。
        """
        rx = x * math.cos(self.rotation) - y * math.sin(self.rotation)
        ry = x * math.sin(self.rotation) + y * math.cos(self.rotation)
        sx = rx * self.scale_factor
        sy = ry * self.scale_factor
        cx = sx + self.offset_x
        cy = -sy + self.offset_y
        return (cx, cy)

    def canvas_to_model(self, cx, cy):
        """
        キャンバス座標 (cx, cy) をモデル座標に変換する。
        """
        x1 = cx - self.offset_x
        y1 = -(cy - self.offset_y)
        if self.scale_factor == 0:
            self.scale_factor = 1
        rx = x1 / self.scale_factor
        ry = y1 / self.scale_factor
        r = -self.rotation
        cosr = math.cos(r)
        sinr = math.sin(r)
        mx = rx * cosr - ry * sinr
        my = rx * sinr + ry * cosr
        return (mx, my)

    # ------------- マウス操作関連 -------------
    def on_left_click(self, e):
        """
        左クリックでモデル座標に変換した点をパスに追加する。
        KDTree も再構築する。
        """
        mx, my = e.x, e.y
        lon, lat = self.canvas_to_model(mx, my)
        self.path.append({"x": lon, "y": lat})
        self.build_path_tree()
        self.event_generate("<<PathUpdated>>", when="tail")
        self.redraw()

    def on_mid_down(self, e):
        self.drag_start_x = e.x
        self.drag_start_y = e.y

    def on_mid_drag(self, e):
        dx = e.x - self.drag_start_x
        dy = e.y - self.drag_start_y
        self.offset_x += dx
        self.offset_y += dy
        self.drag_start_x = e.x
        self.drag_start_y = e.y
        self.redraw()

    def on_mid_up(self, e):
        pass

    def on_mousewheel(self, e):
        factor = 1.1 if e.delta > 0 else 0.9
        cx, cy = e.x, e.y
        self.offset_x = (self.offset_x - cx) * factor + cx
        self.offset_y = (self.offset_y - cy) * factor + cy
        self.scale_factor *= factor
        self.redraw()

    # ------------- シミュレーション関連 -------------
    def update_truck_scale(self):
        """
        トラックの各寸法を度単位に変換する（内部描画用）。
        """
        deg_per_m = self.deg_per_m()
        self.wheelBase_deg = self.wheelBase_m * deg_per_m
        self.frontOverhang_deg = self.frontOverhang_m * deg_per_m
        self.rearOverhang_deg = self.rearOverhang_m * deg_per_m
        self.vehicleWidth_deg = self.vehicleWidth_m * deg_per_m
        self.maxSteeringAngle = deg_to_rad(self.maxSteeringAngle_deg)
        self.vehicleSpeed_deg_s = self.vehicleSpeed_m_s * deg_per_m
        self.lookahead_deg = self.lookahead_m * deg_per_m

    def reset_sim(self):
        """
        シミュレーションの状態を初期化し、再描画する。
        """
        self.running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.update_truck_scale()

        if self.path:
            start_pt = self.path[0]
            total_m = self.frontOverhang_m + self.wheelBase_m + self.rearOverhang_m
            deg_len = total_m * self.deg_per_m()
            if len(self.path) > 1:
                dx = self.path[1]["x"] - start_pt["x"]
                dy = self.path[1]["y"] - start_pt["y"]
                th = math.atan2(dy, dx)
                self.truck_theta = th
            else:
                self.truck_theta = 0
            self.truck_x = start_pt["x"] + deg_len * math.cos(self.truck_theta)
            self.truck_y = start_pt["y"] + deg_len * math.sin(self.truck_theta)
        else:
            self.truck_x = 0
            self.truck_y = 0
            self.truck_theta = 0

        self.truck_velocity = self.vehicleSpeed_deg_s
        self.truck_steering = 0

        self.corner_trajs = [[], [], [], []]
        self.history = []
        self.alert_marker = None

        self.error_integral = 0.0
        self.last_error = 0.0
        self.speed_integral = 0.0
        self.last_speed_error = 0.0

        self.redraw()

    def start_sim(self):
        """
        シミュレーションを開始する。
        パスがなければエラーダイアログを表示する。
        """
        if len(self.path) < 1:
            messagebox.showerror("エラー", "パスが足りません。")
            return
        self.running = True
        self.sim_step()

    def pause_sim(self):
        """シミュレーションを一時停止する。"""
        self.running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None

    def sim_step(self):
        """
        シミュレーションの1ステップ処理。
        ・動的ルックアヘッドの更新
        ・PID/SMC/PurePursuit による制御
        ・バイシクルモデルによる状態更新
        ・報酬関数（1,2,4,5）を計算して記録
        ・終了判定後、次のステップを after() で呼び出す
        """
        if self.master_app:
            self.master_app.last_active_time = time.time()

        if not self.running:
            self.corners_outside = False
            self.event_generate("<<SimFinished>>", when="tail")
            return

        # (1) 動的ルックアヘッド更新
        speed_mps = self.truck_velocity * (1.0 / self.deg_per_m())
        self.lookahead_m = max(3.0, min(15.0, speed_mps * 1.2))
        self.lookahead_deg = self.lookahead_m * self.deg_per_m()

        tgt = self.find_lookahead_target()

        # (2) ステア制御
        if tgt is None:
            combined_steer = 0.0
        else:
            closest_pt = self.find_closest_path_point()
            dx = closest_pt["x"] - self.truck_x
            dy = closest_pt["y"] - self.truck_y
            cte = math.hypot(dx, dy) * math.sin(math.atan2(dy, dx) - self.truck_theta)
            self.cross_track_error = cte

            cte_mag = abs(cte)
            self.pid_kp = 1.2 / (1 + 0.5 * cte_mag)
            self.pid_kd = 0.1 * (1 + cte_mag)
            if cte_mag < 0.5:
                self.error_integral += cte * self.dt
            else:
                self.error_integral *= 0.95
            d_cte = (cte - self.last_error) / self.dt
            pid_out = self.pid_kp * cte + self.pid_ki * self.error_integral + self.pid_kd * d_cte
            self.last_error = cte

            sliding_surf = d_cte + self.smc_lambda * cte
            if abs(sliding_surf) > 1e-3:
                smc_out = -self.smc_eta * math.copysign(1, sliding_surf)
            else:
                smc_out = 0.0

            dx2 = tgt["x"] - self.truck_x
            dy2 = tgt["y"] - self.truck_y
            alpha = math.atan2(dy2, dx2) - self.truck_theta
            alpha = normalize_angle(alpha)
            L = self.wheelBase_deg
            scmd = math.atan2(2.0 * L * math.sin(alpha), self.lookahead_deg)
            scmd = max(-self.maxSteeringAngle, min(self.maxSteeringAngle, scmd))

            combined_steer = scmd + pid_out + smc_out
            combined_steer = max(-self.maxSteeringAngle, min(self.maxSteeringAngle, combined_steer))

        steer_diff = combined_steer - self.truck_steering
        max_diff = self.max_steer_rate * self.dt
        actual_steer = self.truck_steering + max(-max_diff, min(steer_diff, max_diff))
        self.truck_steering = actual_steer

        # (3) 速度制御
        current_speed_mps = self.truck_velocity / self.deg_per_m()
        speed_err = self.vehicleSpeed_m_s - current_speed_mps
        if abs(speed_err) < 0.5:
            self.speed_integral += speed_err * self.dt
        else:
            self.speed_integral = 0
        d_spd = (speed_err - self.last_speed_error) / self.dt
        accel_cmd = self.speed_kp * speed_err + self.speed_ki * self.speed_integral + self.speed_kd * d_spd
        self.last_speed_error = speed_err

        if accel_cmd > 0:
            accel = min(self.max_accel, accel_cmd)
        else:
            accel = max(-self.max_brake, accel_cmd)

        aero_drag = 0.5 * self.drag_coeff * 1.225 * (current_speed_mps ** 2)
        rolling = self.roll_resist * self.vehicle_mass * 9.81
        total_force = (self.vehicle_mass * accel) - aero_drag - rolling
        actual_accel = total_force / self.vehicle_mass

        slip_angle = math.atan2(self.wheelBase_m * math.tan(self.truck_steering), self.wheelBase_m)
        B = 10.0
        C = 1.9
        D = 1.0
        E = 0.97
        pacejka_out = D * math.sin(C * math.atan(B * slip_angle - E * (B * slip_angle - math.atan(B * slip_angle))))
        Fy = pacejka_out * 1000
        self.acceleration = (Fy - rolling) / self.vehicle_mass
        actual_accel += self.acceleration

        new_speed_mps = current_speed_mps + actual_accel * self.dt
        if new_speed_mps < 0:
            new_speed_mps = 0
        self.truck_velocity = new_speed_mps * self.deg_per_m()

        if self.truck_velocity < 1e-6:
            slip = 0
        else:
            slip = math.atan((self.truck_velocity * math.tan(self.truck_steering)) / (self.truck_velocity + 0.1))
        eff_steer = self.truck_steering - slip

        L = self.wheelBase_deg
        if abs(L) > 1e-9:
            ang_v = (self.truck_velocity / L) * math.tan(eff_steer)
        else:
            ang_v = 0
        self.truck_theta += ang_v * self.dt
        self.truck_theta = normalize_angle(self.truck_theta)

        vx = self.truck_velocity * self.dt * math.cos(self.truck_theta)
        vy = self.truck_velocity * self.dt * math.sin(self.truck_theta)
        self.truck_x += vx
        self.truck_y += vy

        # 状態および車両四隅のログ更新
        self.update_vehicle_state()

        # ===== ここから報酬関数の計算 =====
        # 各重み（チューニング必要）
        alpha_weight = 1.0   # クロストラック誤差（二乗）重み
        beta_weight = 0.5    # ヘディング誤差（二乗）重み
        delta_weight = 0.1   # 制御入力（舵角）の絶対値ペナルティ重み
        bonus = 10.0         # 目標到達時ボーナス

        # (1) クロストラック誤差
        cte_val = abs(self.cross_track_error) if hasattr(self, "cross_track_error") else 0.0

        # (2) 進行方向（ヘディング）のずれ
        if tgt is not None:
            target_heading = math.atan2(tgt["y"] - self.truck_y, tgt["x"] - self.truck_x)
            heading_error = abs(normalize_angle(target_heading - self.truck_theta))
        else:
            heading_error = 0.0

        # (4) 制御入力のペナルティ（ここでは combined_steer の絶対値）
        control_penalty = abs(combined_steer)

        # 基本報酬（各ペナルティ項の和）
        reward_value = -alpha_weight * (cte_val ** 2) - beta_weight * (heading_error ** 2) - delta_weight * control_penalty
        # ===== ここまで報酬関数の計算 =====

        self.redraw()

        # 終了判定：最終ウェイポイントとの距離
        dist_fin = 9999
        if self.path:
            dist_fin = math.hypot(self.path[-1]["x"] - self.truck_x, self.path[-1]["y"] - self.truck_y)
        # (5) 目標到達ボーナス
        if dist_fin < self.lookahead_deg * 0.3:
            reward_value += bonus
            self.running = False

        # 最後の履歴レコードに報酬を記録
        if self.history:
            self.history[-1]["reward"] = reward_value

        if self.running:
            self.animation_id = self.after(int(self.dt * 1000), self.sim_step)
        else:
            self.corners_outside = False
            self.event_generate("<<SimFinished>>", when="tail")

    def find_lookahead_target(self):
        """
        現在位置から lookahead 距離の円とパスとの交点を求め、
        最適なターゲット点を返す。
        """
        cx, cy = self.truck_x, self.truck_y
        L = self.lookahead_deg
        best = None
        best_d = float("inf")
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]
            pts = self.circle_line_intersect(cx, cy, L, p1, p2)
            if pts:
                for ip in pts:
                    vx = ip["x"] - cx
                    vy = ip["y"] - cy
                    dotf = vx * math.cos(self.truck_theta) + vy * math.sin(self.truck_theta)
                    if dotf < 0:
                        continue
                    dd = math.hypot(vx, vy)
                    if dd < best_d:
                        best_d = dd
                        best = ip
        if not best and self.path:
            best = self.path[-1]
        return best

    def find_closest_path_point(self):
        """
        パス上の現在位置に最も近い点を返す。
        KDTree があればそちらで探索し、なければ線形探索する。
        """
        if self.path_tree is not None:
            dist, idx = self.path_tree.query((self.truck_x, self.truck_y))
            return self.path[idx]
        else:
            best = None
            bestd = float("inf")
            for pt in self.path:
                dd = math.hypot(pt["x"] - self.truck_x, pt["y"] - self.truck_y)
                if dd < bestd:
                    bestd = dd
                    best = pt
            return best

    def circle_line_intersect(self, cx, cy, r, p1, p2):
        """
        円 (中心 (cx, cy), 半径 r) と直線（p1, p2）の交点を求める。
        """
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        a = dx * dx + dy * dy
        b = 2 * (dx * (p1["x"] - cx) + dy * (p1["y"] - cy))
        c = (p1["x"] - cx) ** 2 + (p1["y"] - cy) ** 2 - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sd = math.sqrt(disc)
        t1 = (-b + sd) / (2 * a)
        t2 = (-b - sd) / (2 * a)
        out = []
        for t_ in [t1, t2]:
            if 0 <= t_ <= 1:
                ix = p1["x"] + t_ * dx
                iy = p1["y"] + t_ * dy
                out.append({"x": ix, "y": iy})
        return out if out else None

    def update_vehicle_state(self):
        """
        車両の四隅（ローカル座標）をワールド座標に変換して軌跡に追加する。
        また、クロストラック誤差や速度なども記録する。
        """
        halfW = self.vehicleWidth_deg * 0.5
        totLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {"x": 0, "y": +halfW},
            {"x": 0, "y": -halfW},
            {"x": -totLen, "y": -halfW},
            {"x": -totLen, "y": +halfW}
        ]
        th = self.truck_theta
        for i, c in enumerate(corners_local):
            rx = c["x"] * math.cos(th) - c["y"] * math.sin(th)
            ry = c["x"] * math.sin(th) + c["y"] * math.cos(th)
            wx = self.truck_x + rx
            wy = self.truck_y + ry
            if i < len(self.corner_trajs):
                self.corner_trajs[i].append({"x": wx, "y": wy})
        cte_val = self.cross_track_error if hasattr(self, "cross_track_error") else 0.0
        self.history.append({
            "x": self.truck_x,
            "y": self.truck_y,
            "theta": self.truck_theta,
            "speed": self.truck_velocity * (1.0 / self.deg_per_m()),
            "steering": self.truck_steering,
            "cte": cte_val
        })

    def draw_truck(self):
        """
        トラックの四隅からポリゴンを描画する。
        """
        if not self.history:
            return
        last = self.history[-1]
        tx, ty = last["x"], last["y"]
        th = last["theta"]
        halfW = self.vehicleWidth_deg * 0.5
        totLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {"x": 0, "y": +halfW},
            {"x": 0, "y": -halfW},
            {"x": -totLen, "y": -halfW},
            {"x": -totLen, "y": +halfW}
        ]
        pts = []
        for c in corners_local:
            rx = c["x"] * math.cos(th) - c["y"] * math.sin(th)
            ry = c["x"] * math.sin(th) + c["y"] * math.cos(th)
            wx = tx + rx
            wy = ty + ry
            px, py = self.model_to_canvas(wx, wy)
            pts.append((px, py))
        if len(pts) == 4:
            arr = []
            for (xx, yy) in pts:
                arr.extend([xx, yy])
            self.create_polygon(*arr, outline="black", fill="", width=2)

    def create_vehicle_alert_marker(self):
        """車両周辺に警告マーカーを描画し、一定時間後に削除する。"""
        if self.alert_marker:
            self.delete(self.alert_marker)
            self.alert_marker = None
        px, py = self.model_to_canvas(self.truck_x, self.truck_y)
        self.alert_marker = self.create_oval(px - 15, py - 15, px + 15, py + 15, outline="red", width=3)
        self.after(1000, self.delete_alert_marker)
    
    def delete_alert_marker(self):
        if self.alert_marker is not None:
            self.delete(self.alert_marker)
            self.alert_marker = None


# --------------------------------------------------
# 4. MainApp: バッチ実行、CSV出力、パラメータ最適化など
# --------------------------------------------------
class MainApp:
    """
    GUIアプリケーション本体。
    連続シミュレーション、パラメータ最適化、CSV出力などの機能を提供する。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("PurePursuit + 曲率判定 & ルックアヘッド最適化")

        # メインフレーム作成
        main_fr = ttk.Frame(root)
        main_fr.pack(fill=tk.BOTH, expand=True)

        # トップフレーム（ボタン配置）
        top_fr = ttk.Frame(main_fr, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_fr, text="GeoJSON読み込み", command=self.on_load_geojson).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="連続開始", command=self.start_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="連続停止", command=self.stop_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="結果を見る", command=self.show_results_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="CSV出力", command=self.export_results_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="パラメータ最適化", command=self.optimize_parameters).pack(side=tk.LEFT, padx=5)

        # シミュレーションとパラメータ表示フレーム
        sim_fr = ttk.Frame(main_fr)
        sim_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 左側：GeoJsonTruckCanvas
        self.canvas = GeoJsonTruckCanvas(sim_fr, master_app=self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右側：パラメータ表示フレーム
        params_fr = ttk.Frame(sim_fr, padding=5, relief=tk.GROOVE)
        params_fr.pack(side=tk.RIGHT, fill=tk.Y)

        self.param_labels = {}
        self.optimizable_params = [
            ("PID_kp", "pid_kp"),
            ("PID_ki", "pid_ki"),
            ("PID_kd", "pid_kd"),
            ("SMC_lambda", "smc_lambda"),
            ("SMC_eta", "smc_eta"),
            ("Lookahead_m", "lookahead_m"),
            ("Speed_kp", "speed_kp"),
            ("Speed_ki", "speed_ki"),
            ("Speed_kd", "speed_kd"),
            ("WheelBase_m", "wheelBase_m"),
            ("FrontOverhang_m", "frontOverhang_m"),
            ("RearOverhang_m", "rearOverhang_m"),
            ("VehicleWidth_m", "vehicleWidth_m"),
            ("MaxSteeringAngle_deg", "maxSteeringAngle_deg"),
            ("VehicleSpeed_m_s", "vehicleSpeed_m_s")
        ]
        for display_name, attr_name in self.optimizable_params:
            lbl = ttk.Label(params_fr, text=f"{display_name}: {getattr(self.canvas, attr_name):.4f}")
            lbl.pack(anchor=tk.W)
            self.param_labels[attr_name] = lbl

        self.batch_mode = False
        self.batch_results = []
        self.all_results = []
        self.last_active_time = time.time()
        self.sim_start_time = None

        self.canvas.bind("<<SimFinished>>", self.on_sim_finished)
        self.root.after(1000, self.check_inactivity)

    # ------------- 30秒停止検知 -------------
    def check_inactivity(self):
        now = time.time()
        if now - self.last_active_time > 30:
            scenario = self.create_scenario(timeout=True)
            self.batch_results.append(scenario)
            self.all_results.append(scenario)
            self.canvas.pause_sim()
            self.randomize_road_widths()
            self.randomize_vehicle_params()
            self.pick_random_path()
            self.canvas.reset_sim()
            self.canvas.start_sim()
            self.last_active_time = time.time()

        if self.batch_mode and self.sim_start_time is not None:
            elapsed = now - self.sim_start_time
            if elapsed > 30:
                self.canvas.pause_sim()
                scenario = self.create_scenario(timeout=True)
                self.batch_results.append(scenario)
                self.all_results.append(scenario)
                self.optimize_parameters()
                self.run_batch_step()
                self.last_active_time = time.time()
                self.sim_start_time = time.time()

        self.root.after(1000, self.check_inactivity)

    # ------------- ボタン操作 -------------
    def on_load_geojson(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All", "*.*")])
        if not fp:
            return
        self.canvas.load_geojson(fp)

    def start_batch(self):
        if not self.canvas.geojson_data:
            messagebox.showerror("GeoJSONなし", "先にGeoJSONを読み込んでください。")
            return
        self.batch_mode = True
        self.batch_results.clear()
        self.all_results.clear()
        self.run_batch_step()

    def stop_batch(self):
        self.batch_mode = False
        self.canvas.pause_sim()
        messagebox.showinfo("停止", "連続モードを終了しました。")

    def run_batch_step(self):
        self.sim_start_time = time.time()
        self.randomize_road_widths()
        self.randomize_vehicle_params()
        self.pick_random_path()
        self.canvas.reset_sim()
        self.canvas.start_sim()

    def on_sim_finished(self, event):
        scenario = self.create_scenario()
        self.batch_results.append(scenario)
        self.all_results.append(scenario)
        if self.batch_mode:
            self.run_batch_step()

    def show_results_list(self):
        if not self.all_results:
            messagebox.showinfo("結果なし", "まだ実行していません。")
            return

        win = tk.Toplevel(self.root)
        win.title("連続実行結果一覧")

        tree = ttk.Treeview(win, columns=("time", "out", "mismatch"), show="headings", height=10)
        tree.heading("time", text="Time")
        tree.heading("out", text="CornersOut")
        tree.heading("mismatch", text="MismatchVal")
        tree.column("time", width=150)
        tree.column("out", width=80)
        tree.column("mismatch", width=100)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        for i, scen in enumerate(self.all_results):
            tstr = scen["time"]
            out = scen["corners_outside"]
            mmv = scen.get("mismatch_value", 0.0)
            tree.insert("", tk.END, values=(tstr, out, f"{mmv:.4f}"))

        def on_dbl(e):
            sel = tree.selection()
            if not sel:
                return
            iid = sel[0]
            idx = tree.index(iid)
            scenario = self.all_results[idx]
            self.show_scenario_detail(scenario)

        tree.bind("<Double-1>", on_dbl)

    def show_scenario_detail(self, scenario):
        dlg = tk.Toplevel(self.root)
        dlg.title("シミュ詳細")

        frm = ttk.Frame(dlg, padding=5)
        frm.pack(fill=tk.BOTH, expand=True)

        lb = tk.Listbox(frm, width=60, height=15)
        lb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        vp = scenario["vehicle_params"]
        lb.insert(tk.END, f"Time: {scenario['time']}")
        lb.insert(tk.END, f"CornersOut: {scenario['corners_outside']}")
        lb.insert(tk.END, f"MismatchValue: {scenario.get('mismatch_value', 0.0):.6f}")
        lb.insert(tk.END, f"RoadWidths: {scenario['road_widths']}")
        lb.insert(tk.END, f"WheelBase_m: {vp['wheelBase_m']:.2f}, FrontOverhang_m: {vp['frontOverhang_m']:.2f}, RearOverhang_m: {vp['rearOverhang_m']:.2f}")
        lb.insert(tk.END, f"VehicleWidth_m: {vp['vehicleWidth_m']:.2f}")
        lb.insert(tk.END, f"MaxSteeringAngle_deg: {vp['maxSteeringAngle_deg']:.2f}, VehicleSpeed_m_s: {vp['vehicleSpeed_m_s']:.2f}")
        lb.insert(tk.END, f"PID_kp/ki/kd= {vp['pid_kp']:.2f}/{vp['pid_ki']:.2f}/{vp['pid_kd']:.2f}")
        lb.insert(tk.END, f"SMC_lambda/eta= {vp['smc_lambda']:.2f}/{vp['smc_eta']:.2f}")
        lb.insert(tk.END, f"Lookahead_m: {vp['lookahead_m']:.2f}")
        lb.insert(tk.END, f"Speed_kp/ki/kd= {vp['speed_kp']:.2f}/{vp['speed_ki']:.2f}/{vp['speed_kd']:.2f}")
        lb.insert(tk.END, f"HistoryLen= {len(scenario['history'])}")

        def plot_history():
            hist = scenario["history"]
            if not hist:
                messagebox.showinfo("No Hist", "履歴がありません。")
                return
            fig, ax = plt.subplots()
            hx = [h["x"] for h in hist]
            hy = [h["y"] for h in hist]
            ax.plot(hx, hy, "r-", label="Trajectory")
            px = [p["x"] for p in scenario["path"]]
            py = [p["y"] for p in scenario["path"]]
            ax.plot(px, py, "b--", label="Reference Path")
            ax.set_aspect("equal")
            ax.legend()
            plt.show()

        btn = ttk.Button(frm, text="軌跡をプロット", command=plot_history)
        btn.pack(side=tk.TOP, pady=5)

    # ------------- ランダム設定関連 -------------
    def randomize_road_widths(self):
        feats = self.canvas.geojson_data.get("features", [])
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                props = ft.setdefault("properties", {})
                props["roadWidthM"] = random.uniform(1.5, 6.0)

    def randomize_vehicle_params(self):
        self.canvas.wheelBase_m = random.uniform(3.0, 6.0)
        self.canvas.frontOverhang_m = random.uniform(0.5, 2.0)
        self.canvas.rearOverhang_m = random.uniform(0.5, 2.0)
        self.canvas.vehicleWidth_m = random.uniform(2.0, 3.0)
        self.canvas.maxSteeringAngle_deg = random.uniform(30, 45)
        self.canvas.vehicleSpeed_m_s = random.uniform(2.0, 8.0)
        self.canvas.lookahead_m = random.uniform(5.0, 15.0)

        self.canvas.pid_kp = random.uniform(0.5, 1.5)
        self.canvas.pid_ki = random.uniform(0.0, 0.2)
        self.canvas.pid_kd = random.uniform(0.0, 0.1)
        self.canvas.smc_lambda = random.uniform(0.3, 0.8)
        self.canvas.smc_eta = random.uniform(0.1, 0.5)

        self.canvas.speed_kp = random.uniform(0.5, 2.0)
        self.canvas.speed_ki = random.uniform(0.0, 0.2)
        self.canvas.speed_kd = random.uniform(0.0, 0.1)

        self.update_param_display()

    def pick_random_path(self, npts=9):
        feats = self.canvas.geojson_data.get("features", [])
        lines = []
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if len(coords) >= npts:
                    lines.append(coords)
        if not lines:
            arr = [(0, 0)]
            for _ in range(npts - 1):
                arr.append((arr[-1][0] + random.uniform(0.001, 0.01),
                            arr[-1][1] + random.uniform(0.001, 0.01)))
            self.canvas.path = [{"x": p[0], "y": p[1]} for p in arr]
            self.canvas.build_path_tree()
            return
        line = random.choice(lines)
        max_start = len(line) - npts
        st = random.randint(0, max_start)
        subset = line[st: st + npts]
        self.canvas.path = [{"x": p[0], "y": p[1]} for p in subset]
        self.canvas.build_path_tree()

    # ------------- パラメータ最適化関連 -------------
    def optimize_parameters(self):
        """
        曲率不一致量（mismatch_value）を最小化するように、
        PID, SMC, Lookahead のパラメータを最適化する。
        """
        def objective(params):
            self.canvas.pid_kp = params[0]
            self.canvas.pid_ki = params[1]
            self.canvas.pid_kd = params[2]
            self.canvas.smc_lambda = params[3]
            self.canvas.smc_eta = params[4]
            self.canvas.lookahead_m = params[5]

            self.canvas.update_truck_scale()
            self.batch_results.clear()
            self.run_batch_step()

            if not self.batch_results:
                return 9999.0
            scen = self.batch_results[-1]
            mismatch = scen.get("mismatch_value", 9999.0)
            return mismatch

        initial_guess = [0.8, 0.01, 0.05, 0.5, 0.2, 10.0]
        bounds = [
            (0.5, 1.5),
            (0.0, 0.2),
            (0.0, 0.1),
            (0.3, 0.8),
            (0.1, 0.5),
            (5.0, 15.0)
        ]
        result = minimize(objective, initial_guess, bounds=bounds, method='Powell')
        best_params = result.x
        self.update_param_display(best_params)

    def run_batch_step(self):
        self.sim_start_time = time.time()
        self.randomize_road_widths()
        self.randomize_vehicle_params()
        self.pick_random_path()
        self.canvas.reset_sim()
        self.canvas.start_sim()

    def create_scenario(self, timeout=False):
        """
        現在のシミュレーション結果からシナリオ（辞書）を生成する。
        """
        now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feats = self.canvas.geojson_data.get("features", [])
        roads = []
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                props = ft.get("properties", {})
                roads.append(props.get("roadWidthM", None))

        path_points = [(p["x"], p["y"]) for p in self.canvas.path]
        corner_trajs_points = []
        for corner_list in self.canvas.corner_trajs:
            cpts = [(pt["x"], pt["y"]) for pt in corner_list]
            corner_trajs_points.append(cpts)

        deg_per_m = self.canvas.deg_per_m()
        mismatch_value = overall_curvature_mismatch(path_points, corner_trajs_points, deg_per_m)
        threshold = 0.05 * (deg_per_m ** 2)
        corners_outside = (mismatch_value > threshold)

        scenario = {
            "time": now_s,
            "vehicle_params": {
                "wheelBase_m": self.canvas.wheelBase_m,
                "frontOverhang_m": self.canvas.frontOverhang_m,
                "rearOverhang_m": self.canvas.rearOverhang_m,
                "vehicleWidth_m": self.canvas.vehicleWidth_m,
                "maxSteeringAngle_deg": self.canvas.maxSteeringAngle_deg,
                "vehicleSpeed_m_s": self.canvas.vehicleSpeed_m_s,
                "lookahead_m": self.canvas.lookahead_m,
                "pid_kp": self.canvas.pid_kp,
                "pid_ki": self.canvas.pid_ki,
                "pid_kd": self.canvas.pid_kd,
                "smc_lambda": self.canvas.smc_lambda,
                "smc_eta": self.canvas.smc_eta,
                "speed_kp": self.canvas.speed_kp,
                "speed_ki": self.canvas.speed_ki,
                "speed_kd": self.canvas.speed_kd
            },
            "road_widths": roads,
            "corners_outside": corners_outside,
            "path": self.canvas.path[:],
            "history": self.canvas.history[:],
            "corner_trajs": [c[:] for c in self.canvas.corner_trajs],
            "mismatch_value": mismatch_value,
            "timeout": timeout
        }
        return scenario

    def update_param_display(self, best_params=None):
        """
        パラメータ表示用ラベルを更新する。
        """
        if best_params is not None:
            self.canvas.pid_kp = best_params[0]
            self.canvas.pid_ki = best_params[1]
            self.canvas.pid_kd = best_params[2]
            self.canvas.smc_lambda = best_params[3]
            self.canvas.smc_eta = best_params[4]
            self.canvas.lookahead_m = best_params[5]
            self.canvas.update_truck_scale()

        display_params = {
            "PID_kp": self.canvas.pid_kp,
            "PID_ki": self.canvas.pid_ki,
            "PID_kd": self.canvas.pid_kd,
            "SMC_lambda": self.canvas.smc_lambda,
            "SMC_eta": self.canvas.smc_eta,
            "Lookahead_m": self.canvas.lookahead_m,
            "Speed_kp": self.canvas.speed_kp,
            "Speed_ki": self.canvas.speed_ki,
            "Speed_kd": self.canvas.speed_kd,
            "WheelBase_m": self.canvas.wheelBase_m,
            "FrontOverhang_m": self.canvas.frontOverhang_m,
            "RearOverhang_m": self.canvas.rearOverhang_m,
            "VehicleWidth_m": self.canvas.vehicleWidth_m,
            "MaxSteeringAngle_deg": self.canvas.maxSteeringAngle_deg,
            "VehicleSpeed_m_s": self.canvas.vehicleSpeed_m_s
        }

        for display_name, attr_name in [
            ("PID_kp", "pid_kp"),
            ("PID_ki", "pid_ki"),
            ("PID_kd", "pid_kd"),
            ("SMC_lambda", "smc_lambda"),
            ("SMC_eta", "smc_eta"),
            ("Lookahead_m", "lookahead_m"),
            ("Speed_kp", "speed_kp"),
            ("Speed_ki", "speed_ki"),
            ("Speed_kd", "speed_kd"),
            ("WheelBase_m", "wheelBase_m"),
            ("FrontOverhang_m", "frontOverhang_m"),
            ("RearOverhang_m", "rearOverhang_m"),
            ("VehicleWidth_m", "vehicleWidth_m"),
            ("MaxSteeringAngle_deg", "maxSteeringAngle_deg"),
            ("VehicleSpeed_m_s", "vehicleSpeed_m_s")
        ]:
            if attr_name in self.param_labels:
                self.param_labels[attr_name].config(text=f"{display_name}: {display_params[display_name]:.4f}")

    def export_results_csv(self):
        """
        シミュレーション結果をCSV形式で出力する。
        """
        if not self.all_results:
            messagebox.showinfo("情報", "出力データがありません。")
            return

        fp = filedialog.asksaveasfilename(
            title="CSVファイル名を指定",
            defaultextension=".csv",
            filetypes=[("CSVファイル", "*.csv"), ("All Files", "*.*")]
        )
        if not fp:
            return

        header = [
            "time",
            "corners_outside",
            "mismatch_value",
            "road_widths",
            "wheelBase_m",
            "frontOverhang_m",
            "rearOverhang_m",
            "vehicleWidth_m",
            "maxSteeringAngle_deg",
            "vehicleSpeed_m_s",
            "lookahead_m",
            "pid_kp",
            "pid_ki",
            "pid_kd",
            "smc_lambda",
            "smc_eta",
            "speed_kp",
            "speed_ki",
            "speed_kd",
            "path_points",
            "speed_history",
            "corners_trajectory"
        ]

        try:
            with open(fp, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for scenario in self.all_results:
                    time_ = scenario["time"]
                    corners_outside = scenario["corners_outside"]
                    mismatch_val = scenario.get("mismatch_value", 0.0)
                    road_widths = scenario["road_widths"]

                    vp = scenario["vehicle_params"]
                    wb = vp["wheelBase_m"]
                    foh = vp["frontOverhang_m"]
                    roh = vp["rearOverhang_m"]
                    w = vp["vehicleWidth_m"]
                    sa = vp["maxSteeringAngle_deg"]
                    spd = vp["vehicleSpeed_m_s"]
                    lhad = vp["lookahead_m"]
                    kp = vp["pid_kp"]
                    ki = vp["pid_ki"]
                    kd = vp["pid_kd"]
                    lam = vp["smc_lambda"]
                    eta = vp["smc_eta"]
                    skp = vp["speed_kp"]
                    ski = vp["speed_ki"]
                    skd = vp["speed_kd"]

                    path_str = ";".join(f"{p['x']:.6f},{p['y']:.6f}" for p in scenario["path"])
                    speed_str = ";".join(f"{h['speed']:.3f}" for h in scenario["history"])

                    corners_str_list = []
                    corner_trajs = scenario["corner_trajs"]
                    for i in range(4):
                        coords_str = ";".join(f"{pt['x']:.6f},{pt['y']:.6f}" for pt in corner_trajs[i])
                        corners_str_list.append(f"corner{i}:[{coords_str}]")
                    corners_str = "|".join(corners_str_list)

                    road_str = ";".join(f"{rw}" for rw in road_widths if rw is not None)

                    row = [
                        time_,
                        corners_outside,
                        f"{mismatch_val:.6f}",
                        road_str,
                        wb,
                        foh,
                        roh,
                        w,
                        sa,
                        spd,
                        lhad,
                        kp,
                        ki,
                        kd,
                        lam,
                        eta,
                        skp,
                        ski,
                        skd,
                        path_str,
                        speed_str,
                        corners_str
                    ]
                    writer.writerow(row)
            messagebox.showinfo("完了", f"CSV出力が完了しました:\n{fp}")
        except Exception as e:
            messagebox.showerror("エラー", f"CSV書き出しに失敗: {e}")

def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
