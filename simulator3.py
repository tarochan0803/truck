import math
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from shapely.geometry import LineString, Point  # Shapely を利用
import numpy as np
from scipy.spatial import KDTree
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

class GeoJsonTruckCanvas(tk.Canvas):
    """
    - GeoJSON(LineString)の roadWidthM(メートル) を読み込み、
      シミュレーション終了時に、トラックの四隅が対象道路のバッファポリゴン内に含まれているか（厳密チェック）を
      Shapely を用いて判定します。
      ※スタートからの移動距離が車両全長の50%以上の場合のみチェック対象とします。
    - 距離計算は「度」単位で行い、各種パラメータは必要に応じて度⇔メートル変換します。
    - Pure Pursuit＋バイシクルモデルによるトップビューシミュレーション。
    - ズーム／回転／パンは純粋な2Dで実装。
    - 道路リスト常時表示、選択時ハイライト、パラメータ編集可能。
    """
    def __init__(self, parent, width=800, height=600, bg="white"):
        super().__init__(parent, width=width, height=height, bg=bg)
        self.pack(fill=tk.BOTH, expand=True)

        # GeoJSON関連
        self.geojson_data = None
        self.bounds = None
        self.geo_points = []
        self.selected_road_featID = None

        # シミュレーション終了時のチェック用
        self.last_alerted_truck_pos = None
        self.road_alert = False

        # 表示変換パラメータ
        self.margin = 20
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.rotation = 0.0

        # 中ボタンドラッグ
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.line_item_to_feature = {}

        # トラック関連（単位: メートル）
        self.wheelBase_m = 4.0
        self.frontOverhang_m = 1.0
        self.rearOverhang_m = 1.0
        self.vehicleWidth_m = 2.5
        self.maxSteeringAngle_deg = 45
        self.vehicleSpeed_m_s = 5.0
        self.lookahead_m = 10.0

        # 上記を度に変換した値（内部計算用）
        self.wheelBase_deg = 0
        self.frontOverhang_deg = 0
        self.rearOverhang_deg = 0
        self.vehicleWidth_deg = 0
        self.maxSteeringAngle = 0
        self.vehicleSpeed_deg_s = 0
        self.lookahead_deg = 0

        # パス＆シミュレーション関連
        self.path = []
        self.running = False
        self.animation_id = None
        self.dt = 0.1

        # トラック状態（モデル座標、度単位）
        self.truck_x = 0
        self.truck_y = 0
        self.truck_theta = 0
        self.truck_velocity = 0
        self.truck_steering = 0

        self.corner_trajs = [[], [], [], []]
        self.history = []

        # 追加パラメータ（物理モデル関連）
        self.max_steer_rate = deg_to_rad(30.0)  # 最大ステアリング角速度 [rad/s]
        self.vehicle_mass = 5000.0             # 車両重量 [kg]
        self.max_accel = 0.3                   # 最大加速度 [m/s²]
        self.max_brake = 3.0                   # 最大減速度 [m/s²]
        self.drag_coeff = 0.35                 # 空気抵抗係数
        self.roll_resist = 0.015               # ローリング抵抗係数
        self.cornering_stiff = 100.0           # コーナリング剛性 [N/deg]（未使用の例）

        # ------------------ PID制御パラメータ ------------------
        # 横方向誤差（CTE）用PID制御
        self.pid_kp = 0.8      # 比例ゲイン
        self.pid_ki = 0.01     # 積分ゲイン
        self.pid_kd = 0.05     # 微分ゲイン
        self.error_integral = 0.0
        self.last_error = 0.0

        # スライディングモード制御（SMC）パラメータ
        self.smc_lambda = 0.5  # スライディングサーフェス係数
        self.smc_eta = 0.2     # 不感帯幅

        # 速度制御用PID制御
        self.speed_kp = 1.0    # 比例ゲイン
        self.speed_ki = 0.1    # 積分ゲイン
        self.speed_kd = 0.05   # 微分ゲイン
        self.speed_integral = 0.0
        self.last_speed_error = 0.0

        # デバッグ表示用
        self.debug_items = {}

        # イベントバインド
        self.bind("<Button-1>", self.on_left_click)
        self.bind("<ButtonPress-2>", self.on_mid_down)
        self.bind("<B2-Motion>", self.on_mid_drag)
        self.bind("<ButtonRelease-2>", self.on_mid_up)
        self.bind("<MouseWheel>", self.on_mousewheel)

        # 逸脱検知用
        self.alert_marker = None

    # ------------------ GeoJSON読み込み ------------------
    def load_geojson(self, fp):
        with open(fp, "r", encoding="utf-8") as f:
            self.geojson_data = json.load(f)
        self.extract_geo_points()
        self.compute_bounds()
        self.full_view()
        self.redraw()
        self.last_alerted_truck_pos = None
        self.road_alert = False

    def extract_geo_points(self):
        self.geo_points = []
        if not self.geojson_data:
            return
        feats = self.geojson_data.get("features", [])
        for feat in feats:
            geom = feat["geometry"]
            if geom["type"] == "LineString":
                for c_ in geom["coordinates"]:
                    self.geo_points.append((c_[0], c_[1]))
            elif geom["type"] == "Point":
                c_ = geom["coordinates"]
                self.geo_points.append((c_[0], c_[1]))

    def compute_bounds(self):
        if not self.geojson_data:
            self.bounds = (0, 1, 0, 1)
            return
        lons, lats = [], []
        for feat in self.geojson_data.get("features", []):
            geom = feat["geometry"]
            if geom["type"] == "LineString":
                for c_ in geom["coordinates"]:
                    lons.append(c_[0])
                    lats.append(c_[1])
            elif geom["type"] == "Point":
                c_ = feat["geometry"]["coordinates"]
                lons.append(c_[0])
                lats.append(c_[1])
        if lons and lats:
            self.bounds = (min(lons), max(lons), min(lats), max(lats))
        else:
            self.bounds = (0, 1, 0, 1)

    def full_view(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 0 or h <= 0:
            return
        (min_lon, max_lon, min_lat, max_lat) = self.bounds
        dx = max_lon - min_lon
        dy = max_lat - min_lat
        if dx < 1e-9 or dy < 1e-9:
            return
        view_w = w - 2 * self.margin
        view_h = h - 2 * self.margin
        if view_w < 10 or view_h < 10:
            return
        scx = view_w / dx
        scy = view_h / dy
        s = min(scx, scy)
        cx, cy = w / 2, h / 2
        mid_lon = (min_lon + max_lon) * 0.5
        mid_lat = (min_lat + max_lat) * 0.5
        self.scale_factor = s
        self.offset_x = cx - s * mid_lon
        self.offset_y = cy + s * mid_lat

    # ------------------ redraw ------------------
    def redraw(self):
        self.delete("all")
        self.line_item_to_feature.clear()
        if self.geojson_data:
            feats = self.geojson_data.get("features", [])
            for f_idx, feat in enumerate(feats):
                geom = feat["geometry"]
                props = feat.get("properties", {})
                road_m = props.get("roadWidthM", 2.0)
                color_ = props.get("color", "gray")
                if f_idx == self.selected_road_featID:
                    color_ = "orange"
                if f_idx == 0 and self.road_alert:
                    color_ = "red"
                roadDeg = road_m * self.deg_per_m()
                roadPx = self.deg_to_px(roadDeg)
                if f_idx == self.selected_road_featID:
                    roadPx += 2
                if geom["type"] == "LineString":
                    coords = geom["coordinates"]
                    if len(coords) > 1:
                        arr = []
                        for c_ in coords:
                            px, py = self.model_to_canvas(c_[0], c_[1])
                            arr.extend([px, py])
                        lid = self.create_line(*arr, fill=color_, width=roadPx)
                        self.line_item_to_feature[lid] = f_idx
                    for c_ in coords:
                        vx, vy = self.model_to_canvas(c_[0], c_[1])
                        self.create_oval(vx-3, vy-3, vx+3, vy+3, fill="red", outline="")
                elif geom["type"] == "Point":
                    c_ = geom["coordinates"]
                    px, py = self.model_to_canvas(c_[0], c_[1])
                    self.create_oval(px-5, py-5, px+5, py+5, fill="red")
        if len(self.path) > 1:
            arr = []
            for p_ in self.path:
                px, py = self.model_to_canvas(p_['x'], p_['y'])
                arr.extend([px, py])
            self.create_line(*arr, fill="blue", width=2)
        for p_ in self.path:
            px, py = self.model_to_canvas(p_['x'], p_['y'])
            self.create_oval(px-4, py-4, px+4, py+4, fill="blue")
        colz = ["red", "green", "blue", "orange"]
        for i, traj in enumerate(self.corner_trajs):
            if len(traj) < 2:
                continue
            tmp = []
            for wpt in traj:
                sx, sy = self.model_to_canvas(wpt['x'], wpt['y'])
                tmp.extend([sx, sy])
            self.create_line(*tmp, fill=colz[i], dash=(2,2), width=1)
        if self.history:
            self.draw_truck()

        # デバッグ情報の描画
        self.display_debug_info()

    def display_debug_info(self):
        # 既存のデバッグテキストを削除
        for tag in self.debug_items:
            self.delete(tag)
        self.debug_items.clear()

        # CTEとステアリング角、速度誤差の表示
        if hasattr(self, 'cross_track_error'):
            cte_text = f"CTE: {self.cross_track_error:.2f}m"
            self.debug_items['cte'] = self.create_text(50, 30, text=cte_text, tag="debug", anchor="nw", fill="black")
        if hasattr(self, 'combined_steering'):
            steer_text = f"Steer: {math.degrees(self.combined_steering):.2f}°"
            self.debug_items['steer'] = self.create_text(50, 50, text=steer_text, tag="debug", anchor="nw", fill="black")
        if hasattr(self, 'speed_error'):
            speed_text = f"Speed Error: {self.speed_error:.2f} m/s"
            self.debug_items['speed_error'] = self.create_text(50, 70, text=speed_text, tag="debug", anchor="nw", fill="black")

    def deg_per_m(self):
        if not self.bounds:
            return 1e-6
        (min_lon, max_lon, min_lat, max_lat) = self.bounds
        lat_c = (min_lat + max_lat) * 0.5
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(lat_c))
        meter_per_deg = (lat_m + lon_m) * 0.5
        return 1.0 / meter_per_deg

    def deg_to_px(self, deg_val):
        x1, y1 = self.model_to_canvas(0, 0)
        x2, y2 = self.model_to_canvas(deg_val, 0)
        return max(1, math.hypot(x2 - x1, y2 - y1))

    # ------------------ model_to_canvas ------------------
    def model_to_canvas(self, lon, lat):
        cosr = math.cos(self.rotation)
        sinr = math.sin(self.rotation)
        rx = lon * cosr - lat * sinr
        ry = lon * sinr + lat * cosr
        sx = rx * self.scale_factor
        sy = ry * self.scale_factor
        cx = sx + self.offset_x
        cy = -sy + self.offset_y
        return (cx, cy)

    def canvas_to_model(self, cx, cy):
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

    # ------------------ マウス操作 ------------------
    def on_left_click(self, event):
        mx, my = event.x, event.y
        (lon, lat) = self.canvas_to_model(mx, my)
        snap_px = 10
        best = (lon, lat)
        best_d = 999999
        for (gx, gy) in self.geo_points:
            vx, vy = self.model_to_canvas(gx, gy)
            dd = math.hypot(vx - mx, vy - my)
            if dd < snap_px and dd < best_d:
                best_d = dd
                best = (gx, gy)
        self.path.append({'x': best[0], 'y': best[1]})
        # 通知: MainApp に道路リストの更新を依頼
        self.event_generate("<<PathUpdated>>")
        self.redraw()

    def on_mid_down(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_mid_drag(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.offset_x += dx
        self.offset_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.redraw()

    def on_mid_up(self, event):
        pass

    def on_mousewheel(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        cx, cy = event.x, event.y
        self.offset_x = (self.offset_x - cx) * factor + cx
        self.offset_y = (self.offset_y - cy) * factor + cy
        self.scale_factor *= factor
        self.redraw()

    # ------------------ トラック＋シミュレーション ------------------
    def update_truck_scale(self):
        if not self.bounds:
            return
        (min_lon, max_lon, min_lat, max_lat) = self.bounds
        lat_c = (min_lat + max_lat) * 0.5
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(lat_c))
        meter_per_deg = (lat_m + lon_m) * 0.5

        self.wheelBase_deg = self.wheelBase_m * meter_per_deg
        self.frontOverhang_deg = self.frontOverhang_m * meter_per_deg
        self.rearOverhang_deg = self.rearOverhang_m * meter_per_deg
        self.vehicleWidth_deg = self.vehicleWidth_m * meter_per_deg
        self.maxSteeringAngle = deg_to_rad(self.maxSteeringAngle_deg)
        self.vehicleSpeed_deg_s = self.vehicleSpeed_m_s * meter_per_deg
        self.lookahead_deg = self.lookahead_m * meter_per_deg

    def reset_sim(self):
        self.running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.update_truck_scale()
        if len(self.path) > 0:
            start_pt = self.path[0]
            # スタート位置を「トラックのケツ」（後部）とするため
            total_length_m = self.frontOverhang_m + self.wheelBase_m + self.rearOverhang_m
            meter_per_deg = 1.0 / self.deg_per_m()
            total_length_deg = total_length_m * meter_per_deg
            if len(self.path) > 1:
                next_pt = self.path[1]
                dx = next_pt['x'] - start_pt['x']
                dy = next_pt['y'] - start_pt['y']
                theta = math.atan2(dy, dx)
                self.truck_theta = theta
            else:
                self.truck_theta = 0
            self.truck_x = start_pt['x'] + total_length_deg * math.cos(self.truck_theta)
            self.truck_y = start_pt['y'] + total_length_deg * math.sin(self.truck_theta)
        else:
            self.truck_x = 0
            self.truck_y = 0
            self.truck_theta = 0
        self.truck_velocity = self.vehicleSpeed_deg_s
        self.truck_steering = 0
        self.corner_trajs = [[], [], [], []]
        self.history = []
        self.last_alerted_truck_pos = None
        self.redraw()

    def start_sim(self):
        if len(self.path) < 1:
            messagebox.showerror("エラー", "パスを設定してください。")
            return
        self.running = True
        self.sim_step()

    def pause_sim(self):
        self.running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None

    def sim_step(self):
        """
        Pure Pursuit + PID制御 + スライディングモード制御（SMC） + バイシクルモデル + 詳細物理モデルを用いたシミュレーションステップ
        """
        # 走行中でなければ終了チェックのみ
        if not self.running:
            self.check_vehicle_fit(during_sim=False)
            return

        # ルックアヘッド先を探索
        tgt = self.find_lookahead_target()
        if not tgt:
            # 目標がなければ終了
            self.running = False
            self.check_vehicle_fit(during_sim=False)
            return

        # 横方向誤差計算（PID用）
        closest_pt = self.find_closest_path_point()
        dx = closest_pt['x'] - self.truck_x
        dy = closest_pt['y'] - self.truck_y
        path_heading = math.atan2(dy, dx)
        # CTEはトラックの向きと経路の角度の差に基づく
        cross_track_error = math.hypot(dx, dy) * math.sin(math.atan2(dy, dx) - self.truck_theta)

        self.cross_track_error = cross_track_error  # デバッグ用

        # PID制御（CTE用）
        self.error_integral += cross_track_error * self.dt
        error_derivative = (cross_track_error - self.last_error) / self.dt
        pid_output = (self.pid_kp * cross_track_error +
                      self.pid_ki * self.error_integral +
                      self.pid_kd * error_derivative)
        self.last_error = cross_track_error

        # スライディングモード制御（SMC）
        sliding_surface = error_derivative + self.smc_lambda * cross_track_error
        smc_output = -self.smc_eta * math.copysign(1, sliding_surface) if abs(sliding_surface) > 1e-3 else 0

        # 既存のPure Pursuit計算
        dx_pu = tgt['x'] - self.truck_x
        dy_pu = tgt['y'] - self.truck_y
        alpha = math.atan2(dy_pu, dx_pu) - self.truck_theta
        alpha = normalize_angle(alpha)

        L = self.wheelBase_deg
        scmd = math.atan2(2 * L * math.sin(alpha), self.lookahead_deg)
        scmd = max(-self.maxSteeringAngle, min(self.maxSteeringAngle, scmd))

        # 複合制御出力
        combined_steering = scmd + pid_output + smc_output
        self.combined_steering = combined_steering  # デバッグ用
        combined_steering = max(-self.maxSteeringAngle, 
                                min(self.maxSteeringAngle, combined_steering))

        # 既存のステアリング制御を置換
        steer_diff = combined_steering - self.truck_steering
        max_allow_diff = self.max_steer_rate * self.dt
        actual_steer = self.truck_steering + max(-max_allow_diff, min(steer_diff, max_allow_diff))
        self.truck_steering = actual_steer

        # 速度制御（PID拡張）
        current_speed_mps = self.truck_velocity * (1.0 / self.deg_per_m())
        speed_error = self.vehicleSpeed_m_s - current_speed_mps
        self.speed_error = speed_error  # デバッグ用

        # 積分項のアンチワインドアップ
        if abs(speed_error) < 0.5:
            self.speed_integral += speed_error * self.dt
        else:
            self.speed_integral = 0

        speed_derivative = (speed_error - self.last_speed_error) / self.dt
        accel_cmd = (self.speed_kp * speed_error +
                     self.speed_ki * self.speed_integral +
                     self.speed_kd * speed_derivative)
        self.last_speed_error = speed_error

        # 制限をかける
        if accel_cmd > 0:
            accel = min(self.max_accel, accel_cmd)
        else:
            accel = max(-self.max_brake, accel_cmd)

        # 抵抗計算
        aero_drag = 0.5 * self.drag_coeff * 1.225 * (current_speed_mps**2)
        rolling_resist = self.roll_resist * self.vehicle_mass * 9.81

        # 加速度計算
        total_force = (self.vehicle_mass * accel) - aero_drag - rolling_resist
        actual_accel = total_force / self.vehicle_mass

        # 詳細物理計算（簡易 Pacejka タイヤモデル）
        Fy = self.calculate_lateral_force()

        # 力のバランスを更新（ここでは Fy を加速度に反映）
        # 実際の物理モデルでは Fy はトルクや他の力に影響するが、簡略化
        self.acceleration = (Fy - self.roll_resist * self.vehicle_mass * 9.81) / self.vehicle_mass
        actual_accel += self.acceleration  # 簡略化された反映

        new_speed_mps = current_speed_mps + actual_accel * self.dt
        new_speed_mps = max(0.0, new_speed_mps)  # 速度は負にならない
        self.truck_velocity = new_speed_mps * self.deg_per_m()

        # スリップアングルの簡易導入（例: ゼロ速度対策）
        if self.truck_velocity < 1e-6:
            slip_angle = 0.0
        else:
            # (本来はコーナリングスティッフネス等で計算するが、サンプルとして単純化)
            slip_angle = math.atan(
                (self.truck_velocity * math.tan(self.truck_steering))
                / (self.truck_velocity + 0.1)
            )

        effective_angle = self.truck_steering - slip_angle

        # 進行方向更新 (バイシクルモデル)
        ang_vel = (self.truck_velocity / self.wheelBase_deg) * math.tan(effective_angle)
        self.truck_theta += ang_vel * self.dt
        self.truck_theta = normalize_angle(self.truck_theta)

        delta_x = self.truck_velocity * self.dt * math.cos(self.truck_theta)
        delta_y = self.truck_velocity * self.dt * math.sin(self.truck_theta)

        self.truck_x += delta_x
        self.truck_y += delta_y

        # ルックアヘッド距離を速度に応じて可変にしてみる
        # (0.5倍～2倍の範囲で変動)
        self.lookahead_deg = max(
            self.lookahead_m * 0.5,
            min(self.lookahead_m * 2.0, new_speed_mps * 1.5)
        ) * self.deg_per_m()

        # 状態更新＆再描画
        self.update_vehicle_state()
        self.redraw()

        # 逸脱検知と可視化
        if self.check_vehicle_fit(during_sim=True):
            self.create_vehicle_alert_marker()

        # 終了判定（最終目標に近ければ）
        dist_ = math.hypot(self.path[-1]['x'] - self.truck_x,
                           self.path[-1]['y'] - self.truck_y)
        if dist_ < self.lookahead_deg * 0.3:
            self.running = False

        # 次フレーム呼び出し or 終了チェック
        if self.running:
            self.animation_id = self.after(int(self.dt * 1000), self.sim_step)
        else:
            self.check_vehicle_fit(during_sim=False)

    def calculate_lateral_force(self):
        # Pacejkaタイヤモデルを簡略化して実装
        B = 0.27   # 剛性因子
        C = 1.65   # 形状因子
        D = 0.7    # ピーク因子

        lateral_slip = math.tan(self.truck_steering) * self.wheelBase_deg
        Fy = D * math.sin(C * math.atan(B * lateral_slip))  # 横力計算
        return Fy

    def find_lookahead_target(self):
        """
        自車位置 (self.truck_x, self.truck_y) を中心とした半径 self.lookahead_deg の
        円がパスと交差する点を探して、その中で最も自車の前方にある点を返す。
        """
        cx, cy = self.truck_x, self.truck_y
        L = self.lookahead_deg
        best = None
        best_d = 999999
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            inters = self.circle_line_intersect(cx, cy, L, p1, p2)
            if inters:
                for ip in inters:
                    vx = ip['x'] - cx
                    vy = ip['y'] - cy
                    # dotfが正なら前方側にある
                    dotf = vx * math.cos(self.truck_theta) + vy * math.sin(self.truck_theta)
                    if dotf < 0:
                        continue
                    dd = math.hypot(ip['x'] - cx, ip['y'] - cy)
                    if dd < best_d:
                        best_d = dd
                        best = ip
        if not best and len(self.path) > 0:
            best = self.path[-1]
        return best

    def find_closest_path_point(self):
        """
        現在位置に最も近いパス上の点を返す。
        """
        min_dist = float('inf')
        closest_pt = None
        truck_point = Point(self.truck_x, self.truck_y)
        for pt in self.path:
            p = Point(pt['x'], pt['y'])
            dist = truck_point.distance(p)
            if dist < min_dist:
                min_dist = dist
                closest_pt = pt
        return closest_pt

    def circle_line_intersect(self, cx, cy, r, p1, p2):
        """
        中心(cx,cy), 半径rの円と、p1->p2の線分が交差する点を求める
        p1, p2, 戻り値は {"x": ..., "y": ...} の辞書
        """
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        a = dx * dx + dy * dy
        b = 2 * (dx * (p1['x'] - cx) + dy * (p1['y'] - cy))
        c = (p1['x'] - cx)**2 + (p1['y'] - cy)**2 - r*r
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        sd = math.sqrt(disc)
        t1 = (-b + sd) / (2*a)
        t2 = (-b - sd) / (2*a)
        out = []
        for t_ in (t1, t2):
            if 0 <= t_ <= 1:
                ix = p1['x'] + t_ * dx
                iy = p1['y'] + t_ * dy
                out.append({'x': ix, 'y': iy})
        return out if out else None

    # ------------------ 車両適合チェック ------------------
    def check_vehicle_fit(self, during_sim=False):
        """
        シミュレーション終了時（またはduring_sim=Falseの場合）に、
        スタートからの移動距離が車両全長の50%以上の場合、
        トラックの四隅が対象道路（GeoJSON先頭フィーチャ）のバッファポリゴンに
        収まっているかをShapelyでチェックし、結果に応じてメッセージ表示する。
        """
        if not self.geojson_data or not self.path:
            return False

        # チェック対象はスタートからの移動距離が車両全長の50%以上の場合のみ
        start_pt = self.path[0]
        dx = self.truck_x - start_pt['x']
        dy = self.truck_y - start_pt['y']
        dist_deg = math.hypot(dx, dy)
        meter_per_deg = 1.0 / self.deg_per_m()
        dist_m = dist_deg * meter_per_deg
        total_length_m = self.frontOverhang_m + self.wheelBase_m + self.rearOverhang_m
        if dist_m < (total_length_m * 0.5):
            return False

        # 対象道路のLineString取得
        feats = self.geojson_data.get("features", [])
        if not feats:
            return False
        feat = feats[0]
        geom = feat["geometry"]
        if geom["type"] != "LineString":
            return False
        props = feat.setdefault("properties", {})
        roadWidthM = props.get("roadWidthM", 2.0)
        half_width_m = roadWidthM / 2.0
        # バッファ用度の計算
        buffer_deg = half_width_m * self.deg_per_m()

        road_coords = geom["coordinates"]
        road_line = LineString(road_coords)
        road_buffer = road_line.buffer(buffer_deg)

        # トラック四隅の座標をShapely.Point化して check
        truck_corners = []
        halfW = self.vehicleWidth_deg * 0.5
        totalLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {'x': 0, 'y': +halfW},                # Front Right
            {'x': 0, 'y': -halfW},                # Front Left
            {'x': -totalLen, 'y': -halfW},        # Rear Left
            {'x': -totalLen, 'y': +halfW},        # Rear Right
        ]
        th = self.truck_theta
        tx, ty = self.truck_x, self.truck_y
        for c in corners_local:
            rx = c['x'] * math.cos(th) - c['y'] * math.sin(th)
            ry = c['x'] * math.sin(th) + c['y'] * math.cos(th)
            corner_lon = tx + rx
            corner_lat = ty + ry
            truck_corners.append(Point(corner_lon, corner_lat))

        all_inside = all(road_buffer.contains(pt) for pt in truck_corners)
        if all_inside:
            if not during_sim:
                messagebox.showinfo("搬入成功", "搬入成功")
            return False
        else:
            if not during_sim:
                messagebox.showerror("搬入不可", "この車両では搬入できません")
            return True

    # ------------------ トラック描画 ------------------
    def update_vehicle_state(self):
        halfW = self.vehicleWidth_deg * 0.5
        totalLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {'x': 0, 'y': +halfW},
            {'x': 0, 'y': -halfW},
            {'x': -totalLen, 'y': -halfW},
            {'x': -totalLen, 'y': +halfW},
        ]
        th = self.truck_theta
        for i, c in enumerate(corners_local):
            rx = c['x'] * math.cos(th) - c['y'] * math.sin(th)
            ry = c['x'] * math.sin(th) + c['y'] * math.cos(th)
            wx = self.truck_x + rx
            wy = self.truck_y + ry
            if i < len(self.corner_trajs):
                self.corner_trajs[i].append({'x': wx, 'y': wy})
        self.history.append({
            'x': self.truck_x,
            'y': self.truck_y,
            'theta': self.truck_theta,
            'speed': self.truck_velocity * (1.0 / self.deg_per_m()),
            'steering': self.truck_steering,
            'cte': self.cross_track_error if hasattr(self, 'cross_track_error') else 0.0
        })

    def draw_truck(self):
        if not self.history:
            return
        last = self.history[-1]
        tx, ty = last['x'], last['y']
        th = last['theta']
        halfW = self.vehicleWidth_deg * 0.5
        totalLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {'x': 0, 'y': +halfW},
            {'x': 0, 'y': -halfW},
            {'x': -totalLen, 'y': -halfW},
            {'x': -totalLen, 'y': +halfW},
        ]
        pts = []
        for c in corners_local:
            rx = c['x'] * math.cos(th) - c['y'] * math.sin(th)
            ry = c['x'] * math.sin(th) + c['y'] * math.cos(th)
            wx = tx + rx
            wy = ty + ry
            px, py = self.model_to_canvas(wx, wy)
            pts.append((px, py))
        if len(pts) == 4:
            arr = []
            for (xx, yy) in pts:
                arr.extend([xx, yy])
            self.create_polygon(*arr, outline="black", fill="", width=2)

    # ------------------ リアルタイム逸脱検知＆可視化機能 ------------------
    def create_vehicle_alert_marker(self):
        # 既にアラートマーカーが存在する場合は削除
        if self.alert_marker:
            self.delete(self.alert_marker)
            self.alert_marker = None
        tx, ty = self.truck_x, self.truck_y
        px, py = self.model_to_canvas(tx, ty)
        self.alert_marker = self.create_oval(px-15, py-15, px+15, py+15, outline="red", width=3, tag="alert")
        self.after(1000, self.delete_alert_marker)

    def delete_alert_marker(self):
        if self.alert_marker:
            self.delete(self.alert_marker)
            self.alert_marker = None

    # ------------------ 3Dビューア機能（Matplotlib連携） ------------------
    def create_3d_viewer(self):
        self.viewer = ThreeDViewer(self)
        self.viewer.update_plot()

    # ------------------ 動的経路計画機能（RRTアルゴリズム統合） ------------------
    def dynamic_path_planning(self, goal):
        obstacles = self.get_obstacles()
        planner = DynamicPathPlanner(obstacles)
        start = (self.truck_x, self.truck_y)
        path = planner.find_path(start, goal)
        if path:
            self.path = path
            self.event_generate("<<PathUpdated>>")  # 通知: MainApp に道路リストの更新を依頼
            self.redraw()
            messagebox.showinfo("パス計画", "動的経路計画が完了しました。")
        else:
            messagebox.showerror("パス計画", "経路が見つかりませんでした。")

    def get_obstacles(self):
        # GeoJSONの障害物を取得（ここでは簡略化）
        # 実際には道路やその他の障害物を適切に抽出する必要があります
        obstacles = []
        if self.geojson_data:
            feats = self.geojson_data.get("features", [])
            for feat in feats:
                geom = feat["geometry"]
                if geom["type"] == "Point":
                    obstacles.append((geom['coordinates'][0], geom['coordinates'][1]))
        return obstacles

    # ------------------ 詳細物理計算（タイヤモデル追加） ------------------
    # 既に calculate_lateral_force メソッドで実装済み

    # ------------------ シナリオ管理機能（JSONベース） ------------------
    def save_scenario(self, filename):
        scenario = {
            'vehicle_params': {
                'wheelBase_m': self.wheelBase_m,
                'frontOverhang_m': self.frontOverhang_m,
                'rearOverhang_m': self.rearOverhang_m,
                'vehicleWidth_m': self.vehicleWidth_m,
                'maxSteeringAngle_deg': self.maxSteeringAngle_deg,
                'vehicleSpeed_m_s': self.vehicleSpeed_m_s,
                'lookahead_m': self.lookahead_m,
                'max_steer_rate': self.max_steer_rate,
                'vehicle_mass': self.vehicle_mass,
                'max_accel': self.max_accel,
                'max_brake': self.max_brake,
                'pid_kp': self.pid_kp,
                'pid_ki': self.pid_ki,
                'pid_kd': self.pid_kd,
                'smc_lambda': self.smc_lambda,
                'smc_eta': self.smc_eta,
                'speed_kp': self.speed_kp,
                'speed_ki': self.speed_ki,
                'speed_kd': self.speed_kd
            },
            'path': self.path,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(scenario, f, indent=4)
        messagebox.showinfo("シナリオ保存", f"シナリオを保存しました: {filename}")

    def load_scenario(self, filename):
        with open(filename, 'r') as f:
            scenario = json.load(f)
        vp = scenario.get('vehicle_params', {})
        self.wheelBase_m = vp.get('wheelBase_m', 4.0)
        self.frontOverhang_m = vp.get('frontOverhang_m', 1.0)
        self.rearOverhang_m = vp.get('rearOverhang_m', 1.0)
        self.vehicleWidth_m = vp.get('vehicleWidth_m', 2.5)
        self.maxSteeringAngle_deg = vp.get('maxSteeringAngle_deg', 45)
        self.vehicleSpeed_m_s = vp.get('vehicleSpeed_m_s', 5.0)
        self.lookahead_m = vp.get('lookahead_m', 10.0)
        self.max_steer_rate = vp.get('max_steer_rate', deg_to_rad(30.0))
        self.vehicle_mass = vp.get('vehicle_mass', 5000.0)
        self.max_accel = vp.get('max_accel', 0.3)
        self.max_brake = vp.get('max_brake', 3.0)
        self.pid_kp = vp.get('pid_kp', 0.8)
        self.pid_ki = vp.get('pid_ki', 0.01)
        self.pid_kd = vp.get('pid_kd', 0.05)
        self.smc_lambda = vp.get('smc_lambda', 0.5)
        self.smc_eta = vp.get('smc_eta', 0.2)
        self.speed_kp = vp.get('speed_kp', 1.0)
        self.speed_ki = vp.get('speed_ki', 0.1)
        self.speed_kd = vp.get('speed_kd', 0.05)
        self.path = scenario.get('path', [])
        self.history = scenario.get('history', [])
        self.reset_sim()
        self.redraw()
        messagebox.showinfo("シナリオ読み込み", f"シナリオを読み込みました: {filename}")

    # ------------------ データ分析機能（Matplotlib統合） ------------------
    def show_analytics_dashboard(self):
        if not self.history:
            messagebox.showerror("データ分析", "履歴データがありません。")
            return
        dashboard = AnalyticsDashboard(self.history)
        dashboard.plot_metrics()

class ThreeDViewer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update_plot(self):
        self.ax.clear()
        # 道路データの3Dプロット
        if self.canvas.geojson_data:
            for feat in self.canvas.geojson_data['features']:
                if feat['geometry']['type'] == 'LineString':
                    coords = np.array(feat['geometry']['coordinates'])
                    self.ax.plot(coords[:,0], coords[:,1], zs=0, c='gray')
        # 車両軌跡のプロット
        history = np.array([(p['x'], p['y']) for p in self.canvas.history])
        if len(history) > 0:
            self.ax.plot(history[:,0], history[:,1], zs=0, c='blue')
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_zlabel('Altitude')
        plt.title("3Dビューア")
        plt.tight_layout()
        plt.show()

class DynamicPathPlanner:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        if obstacles:
            self.kdtree = KDTree(obstacles)
        else:
            self.kdtree = None

    def find_path(self, start, goal):
        # 簡易RRT実装
        nodes = [start]
        parents = {start: None}
        for _ in range(1000):
            if self.obstacles:
                # 障害物周辺からランダムポイントを生成するよう修正
                rand_coords = [random.uniform(min(coord), max(coord)) for coord in zip(*self.obstacles)]
                rand = tuple(rand_coords)
            else:
                rand = goal
            nearest = self.find_nearest(nodes, rand)
            new = self.steer(nearest, rand)
            if not self.check_collision(nearest, new):
                nodes.append(new)
                parents[new] = nearest
                if self.distance(new, goal) < 0.1:
                    return self.reconstruct_path(parents, new, goal)
        return None

    def find_nearest(self, nodes, point):
        if self.kdtree:
            distance, index = self.kdtree.query(point)
            return nodes[index]
        else:
            return min(nodes, key=lambda p: self.distance(p, point))

    def steer(self, from_, to, extend_length=0.5):
        theta = math.atan2(to[1] - from_[1], to[0] - from_[0])
        new_x = from_[0] + extend_length * math.cos(theta)
        new_y = from_[1] + extend_length * math.sin(theta)
        return (new_x, new_y)

    def check_collision(self, from_, to):
        # 簡易衝突チェック（直線上に障害物がないか）
        for obs in self.obstacles:
            if self.line_point_distance(from_, to, obs) < 0.2:  # しきい値
                return True
        return False

    def line_point_distance(self, p1, p2, p):
        # 線分p1-p2と点pの距離を計算
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = p
        num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        den = math.hypot(y2 - y1, x2 - x1)
        if den == 0:
            return math.hypot(x0 - x1, y0 - y1)
        return num / den

    def distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def reconstruct_path(self, parents, current, goal):
        path = [{'x': current[0], 'y': current[1]}]
        while parents[current] is not None:
            current = parents[current]
            path.append({'x': current[0], 'y': current[1]})
        path.reverse()
        path.append({'x': goal[0], 'y': goal[1]})
        return path

class ScenarioManager:
    def __init__(self):
        self.scenarios = []

    def add_scenario(self, params, path):
        self.scenarios.append({
            'vehicle': params,
            'path': path,
            'timestamp': datetime.now().isoformat()
        })

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.scenarios, f, indent=4)
        messagebox.showinfo("シナリオ保存", f"シナリオを保存しました: {filename}")

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            self.scenarios = json.load(f)

class AnalyticsDashboard:
    def __init__(self, history):
        self.history = history

    def plot_metrics(self):
        if not self.history:
            messagebox.showerror("データ分析", "履歴データがありません。")
            return
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        axs[0].plot([h['speed'] for h in self.history], label='Speed (m/s)')
        axs[0].set_title('Speed Profile')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Speed (m/s)')
        axs[0].legend()
        
        axs[1].plot([math.degrees(h['steering']) for h in self.history], label='Steering Angle (°)', color='orange')
        axs[1].set_title('Steering Angle')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Steering Angle (°)')
        axs[1].legend()
        
        axs[2].plot([h['cte'] for h in self.history], label='Cross Track Error (m)', color='red')
        axs[2].set_title('Cross Track Error')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('CTE (m)')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("搬入可否チェック（シミュ終了時） + 厳密チェック（Shapely採用）")
        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)

        # GeoJSON領域
        load_fr = ttk.Frame(top_fr)
        load_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_fr, text="GeoJSON読み込み", command=self.on_load).pack(fill=tk.X, pady=2)
        ttk.Button(load_fr, text="全体表示", command=self.on_fit).pack(fill=tk.X, pady=2)

        # 回転領域
        rot_fr = ttk.LabelFrame(top_fr, text="回転", padding=5)
        rot_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_fr, text="←(-10°)", command=lambda: self.on_rotate(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_fr, text="→(+10°)", command=lambda: self.on_rotate(10)).pack(side=tk.LEFT, padx=2)

        sep = ttk.Separator(top_fr, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # トラックパラメータ用のフレームをインスタンス変数として保持
        self.param_fr = ttk.LabelFrame(top_fr, text="トラックパラメータ", padding=5)
        self.param_fr.pack(side=tk.LEFT, padx=5)
        self.entries = {}
        self.param_frames = []  # 各パラメータのフレームをリストで保持

        def addparam(lbl, defv, key):
            rfr = ttk.Frame(self.param_fr)
            rfr.pack(anchor="w", pady=2)
            ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
            e = ttk.Entry(rfr, width=6)
            e.insert(0, defv)
            e.pack(side=tk.LEFT)
            self.entries[key] = e
            self.param_frames.append(rfr)  # フレームをリストに追加

        addparam("WB(m):", "4.0", "wb")
        addparam("前OH(m):", "1.0", "fo")
        addparam("後OH(m):", "1.0", "ro")
        addparam("幅(m):", "2.5", "w")
        addparam("ステア角(°):", "45", "steer")
        addparam("速度(m/s):", "5", "spd")
        addparam("ルック(m):", "10", "look")
        addparam("ステア速度(°/s):", "30", "steer_rate")
        addparam("車両重量(kg):", "5000", "mass")
        addparam("最大加速度:", "0.3", "max_accel")
        addparam("最大減速度:", "3.0", "max_brake")

        # インタラクティブパラメータ調整用フレームをインスタンス変数として保持
        self.control_fr = ttk.LabelFrame(top_fr, text="制御パラメータ", padding=5)
        self.control_fr.pack(side=tk.LEFT, padx=5)
        self.control_frames = []  # 各スライダーのフレームをリストで保持

        # PIDパラメータ用スライダーをインスタンス変数として保持
        self.pid_kp_var = tk.DoubleVar(value=0.8)
        self.pid_ki_var = tk.DoubleVar(value=0.01)
        self.pid_kd_var = tk.DoubleVar(value=0.05)
        self.smc_lambda_var = tk.DoubleVar(value=0.5)
        self.smc_eta_var = tk.DoubleVar(value=0.2)
        self.speed_kp_var = tk.DoubleVar(value=1.0)
        self.speed_ki_var = tk.DoubleVar(value=0.1)
        self.speed_kd_var = tk.DoubleVar(value=0.05)

        # スライダーの追加とフレームの保持
        self.control_frames.append(self.add_slider(self.control_fr, "PID KP", 0.0, 2.0, self.pid_kp_var, self.update_pid_params))
        self.control_frames.append(self.add_slider(self.control_fr, "PID KI", 0.0, 1.0, self.pid_ki_var, self.update_pid_params))
        self.control_frames.append(self.add_slider(self.control_fr, "PID KD", 0.0, 1.0, self.pid_kd_var, self.update_pid_params))
        self.control_frames.append(self.add_slider(self.control_fr, "SMC Lambda", 0.0, 1.0, self.smc_lambda_var, self.update_smc_params))
        self.control_frames.append(self.add_slider(self.control_fr, "SMC Eta", 0.0, 1.0, self.smc_eta_var, self.update_smc_params))
        self.control_frames.append(self.add_slider(self.control_fr, "Speed PID KP", 0.0, 2.0, self.speed_kp_var, self.update_speed_pid_params))
        self.control_frames.append(self.add_slider(self.control_fr, "Speed PID KI", 0.0, 1.0, self.speed_ki_var, self.update_speed_pid_params))
        self.control_frames.append(self.add_slider(self.control_fr, "Speed PID KD", 0.0, 1.0, self.speed_kd_var, self.update_speed_pid_params))

        # シミュレーション領域
        sim_fr = ttk.LabelFrame(top_fr, text="シミュレーション", padding=5)
        sim_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_fr, text="リセット", command=self.on_reset).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="スタート", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="一時停止", command=self.on_pause).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="動的経路計画", command=self.on_dynamic_path_planning).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="シナリオ保存", command=self.on_save_scenario).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="シナリオ読み込み", command=self.on_load_scenario).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="データ分析", command=self.on_show_analytics).pack(fill=tk.X, pady=2)

        # 3Dビューアボタン
        ttk.Button(top_fr, text="3D表示", command=self.on_show_3d_view).pack(side=tk.LEFT, padx=5)

        main_fr = ttk.Frame(root)
        main_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = GeoJsonTruckCanvas(main_fr)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 道路リスト領域
        roadlist_fr = ttk.LabelFrame(main_fr, text="道路リスト(選択でハイライト)", padding=5)
        roadlist_fr.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.road_tree = ttk.Treeview(roadlist_fr, columns=("featID", "roadWidthM"), show="headings", height=15)
        self.road_tree.heading("featID", text="ID")
        self.road_tree.heading("roadWidthM", text="幅(m)")
        self.road_tree.column("featID", width=50)
        self.road_tree.column("roadWidthM", width=80)
        self.road_tree.pack(side=tk.TOP, fill=tk.Y, expand=True)

        apply_btn = ttk.Button(roadlist_fr, text="適用", command=self.on_road_apply)
        apply_btn.pack(side=tk.TOP, pady=8)

        self.road_tree.bind("<<TreeviewSelect>>", self.on_road_select)
        self.road_tree.bind("<Double-1>", self.on_road_dblclick)

        # シナリオ管理
        self.scenario_manager = ScenarioManager()

        # トグルボタンの追加
        toggle_btn = ttk.Button(top_fr, text="パラメータ表示/非表示", command=self.toggle_parameters)
        toggle_btn.pack(side=tk.LEFT, padx=5)

        # イベントバインド: パス更新時に道路リストを更新
        self.canvas.bind("<<PathUpdated>>", self.populate_road_list)

    # スライダーを追加し、フレームを返す
    def add_slider(self, parent, label, from_, to, var, callback):
        frame = ttk.Frame(parent)
        frame.pack(pady=5, fill=tk.X)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        scale = tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var, resolution=0.01,
                        command=lambda val, cb=callback: cb())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return frame  # フレーム自体を返す

    def addparam(self, lbl, defv, key):
        rfr = ttk.Frame(self.param_fr)
        rfr.pack(anchor="w", pady=2)
        ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
        e = ttk.Entry(rfr, width=6)
        e.insert(0, defv)
        e.pack(side=tk.LEFT)
        self.entries[key] = e
        self.param_frames.append(rfr)  # フレームをリストに追加

    def update_pid_params(self):
        self.canvas.pid_kp = self.pid_kp_var.get()
        self.canvas.pid_ki = self.pid_ki_var.get()
        self.canvas.pid_kd = self.pid_kd_var.get()

    def update_smc_params(self):
        self.canvas.smc_lambda = self.smc_lambda_var.get()
        self.canvas.smc_eta = self.smc_eta_var.get()

    def update_speed_pid_params(self):
        self.canvas.speed_kp = self.speed_kp_var.get()
        self.canvas.speed_ki = self.speed_ki_var.get()
        self.canvas.speed_kd = self.speed_kd_var.get()

    # パラメータ表示/非表示を切り替えるトグル関数
    def toggle_parameters(self):
        # 制御パラメータフレームの表示・非表示を切り替える
        if self.control_fr.winfo_ismapped():
            self.control_fr.pack_forget()
        else:
            self.control_fr.pack(side=tk.LEFT, padx=5)

        # トラックパラメータフレームの表示・非表示を切り替える
        if self.param_fr.winfo_ismapped():
            self.param_fr.pack_forget()
        else:
            self.param_fr.pack(side=tk.LEFT, padx=5)

    # シミュレーション関連の関数
    def on_reset(self):
        self.apply_params()
        self.canvas.path.clear()
        self.canvas.reset_sim()

    def on_start(self):
        if len(self.canvas.path) < 1:
            messagebox.showerror("エラー", "パスを設定してください。")
            return
        self.apply_params()
        self.canvas.reset_sim()
        self.canvas.start_sim()

    def on_pause(self):
        self.canvas.pause_sim()

    def on_dynamic_path_planning(self):
        if not self.canvas.geojson_data:
            messagebox.showerror("エラー", "GeoJSONデータがロードされていません。")
            return
        # 目標地点をユーザーに選択させる（ダイアログで入力）
        try:
            goal_x = float(simpledialog.askstring("動的経路計画", "目標地点のX座標を入力してください:"))
            goal_y = float(simpledialog.askstring("動的経路計画", "目標地点のY座標を入力してください:"))
            goal = (goal_x, goal_y)
            self.canvas.dynamic_path_planning(goal)
        except (TypeError, ValueError):
            messagebox.showerror("入力エラー", "有効な座標を入力してください。")

    def on_save_scenario(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            self.canvas.save_scenario(filename)

    def on_load_scenario(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if filename:
            self.canvas.load_scenario(filename)

    def on_show_analytics(self):
        self.canvas.show_analytics_dashboard()

    def on_show_3d_view(self):
        self.canvas.create_3d_viewer()

    def apply_params(self):
        try:
            wb = float(self.entries["wb"].get())
            fo = float(self.entries["fo"].get())
            ro = float(self.entries["ro"].get())
            w_ = float(self.entries["w"].get())
            st = float(self.entries["steer"].get())
            spd = float(self.entries["spd"].get())
            lk = float(self.entries["look"].get())
            steer_rate = float(self.entries["steer_rate"].get())
            mass_ = float(self.entries["mass"].get())
            max_accel_ = float(self.entries["max_accel"].get())
            max_brake_ = float(self.entries["max_brake"].get())
        except ValueError:
            messagebox.showerror("入力エラー", "正しい数値を入力してください。")
            return
        self.canvas.wheelBase_m = wb
        self.canvas.frontOverhang_m = fo
        self.canvas.rearOverhang_m = ro
        self.canvas.vehicleWidth_m = w_
        self.canvas.maxSteeringAngle_deg = st
        self.canvas.vehicleSpeed_m_s = spd
        self.canvas.lookahead_m = lk
        self.canvas.max_steer_rate = deg_to_rad(steer_rate)
        self.canvas.vehicle_mass = mass_
        self.canvas.max_accel = max_accel_
        self.canvas.max_brake = max_brake_

    # 道路リスト関連の関数
    def populate_road_list(self, event=None):
        self.road_tree.delete(*self.road_tree.get_children())
        if not self.canvas.geojson_data:
            return
        feats = self.canvas.geojson_data["features"]
        for i, feat in enumerate(feats):
            geom = feat["geometry"]
            if geom["type"] == "LineString":
                props = feat.setdefault("properties", {})
                rw = props.get("roadWidthM", 2.0)
                self.road_tree.insert("", tk.END, values=(i, rw))

    def on_road_select(self, event):
        sel = self.road_tree.selection()
        if not sel:
            return
        iid = sel[0]
        vals = self.road_tree.item(iid, "values")
        featID = int(vals[0])
        self.canvas.selected_road_featID = featID
        self.canvas.redraw()

    def on_road_dblclick(self, event):
        sel = self.road_tree.selection()
        if not sel:
            return
        iid = sel[0]
        vals = self.road_tree.item(iid, "values")
        featID = int(vals[0])
        oldVal = float(vals[1])
        newVal = simpledialog.askfloat("幅編集", f"現在: {oldVal}\n新しい幅(m)を入力", minvalue=0.1)
        if newVal is not None:
            self.road_tree.item(iid, values=(featID, newVal))

    def on_road_apply(self):
        if not self.canvas.geojson_data:
            return
        feats = self.canvas.geojson_data["features"]
        for iid in self.road_tree.get_children():
            vals = self.road_tree.item(iid, "values")
            featID = int(vals[0])
            try:
                wVal = float(vals[1])
            except:
                wVal = 2.0
            props = feats[featID].setdefault("properties", {})
            props["roadWidthM"] = wVal
        self.canvas.selected_road_featID = None
        self.canvas.redraw()
        messagebox.showinfo("OK", "道路幅を更新しました")

    # GeoJSON読み込み関連の関数
    def on_load(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All", "*.*")])
        if fp:
            self.canvas.load_geojson(fp)
            self.populate_road_list()

    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()

    def on_rotate(self, deg_):
        r = math.radians(deg_)
        self.canvas.rotation = normalize_angle(self.canvas.rotation + r)
        self.canvas.redraw()

class ThreeDViewer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update_plot(self):
        self.ax.clear()
        # 道路データの3Dプロット
        if self.canvas.geojson_data:
            for feat in self.canvas.geojson_data['features']:
                if feat['geometry']['type'] == 'LineString':
                    coords = np.array(feat['geometry']['coordinates'])
                    self.ax.plot(coords[:,0], coords[:,1], zs=0, c='gray')
        # 車両軌跡のプロット
        history = np.array([(p['x'], p['y']) for p in self.canvas.history])
        if len(history) > 0:
            self.ax.plot(history[:,0], history[:,1], zs=0, c='blue')
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_zlabel('Altitude')
        plt.title("3Dビューア")
        plt.tight_layout()
        plt.show()

class DynamicPathPlanner:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        if obstacles:
            self.kdtree = KDTree(obstacles)
        else:
            self.kdtree = None

    def find_path(self, start, goal):
        # 簡易RRT実装
        nodes = [start]
        parents = {start: None}
        for _ in range(1000):
            if self.obstacles:
                # 障害物周辺からランダムポイントを生成するよう修正
                rand_coords = [random.uniform(min(coord), max(coord)) for coord in zip(*self.obstacles)]
                rand = tuple(rand_coords)
            else:
                rand = goal
            nearest = self.find_nearest(nodes, rand)
            new = self.steer(nearest, rand)
            if not self.check_collision(nearest, new):
                nodes.append(new)
                parents[new] = nearest
                if self.distance(new, goal) < 0.1:
                    return self.reconstruct_path(parents, new, goal)
        return None

    def find_nearest(self, nodes, point):
        if self.kdtree:
            distance, index = self.kdtree.query(point)
            return nodes[index]
        else:
            return min(nodes, key=lambda p: self.distance(p, point))

    def steer(self, from_, to, extend_length=0.5):
        theta = math.atan2(to[1] - from_[1], to[0] - from_[0])
        new_x = from_[0] + extend_length * math.cos(theta)
        new_y = from_[1] + extend_length * math.sin(theta)
        return (new_x, new_y)

    def check_collision(self, from_, to):
        # 簡易衝突チェック（直線上に障害物がないか）
        for obs in self.obstacles:
            if self.line_point_distance(from_, to, obs) < 0.2:  # しきい値
                return True
        return False

    def line_point_distance(self, p1, p2, p):
        # 線分p1-p2と点pの距離を計算
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = p
        num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        den = math.hypot(y2 - y1, x2 - x1)
        if den == 0:
            return math.hypot(x0 - x1, y0 - y1)
        return num / den

    def distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def reconstruct_path(self, parents, current, goal):
        path = [{'x': current[0], 'y': current[1]}]
        while parents[current] is not None:
            current = parents[current]
            path.append({'x': current[0], 'y': current[1]})
        path.reverse()
        path.append({'x': goal[0], 'y': goal[1]})
        return path

class ScenarioManager:
    def __init__(self):
        self.scenarios = []

    def add_scenario(self, params, path):
        self.scenarios.append({
            'vehicle': params,
            'path': path,
            'timestamp': datetime.now().isoformat()
        })

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.scenarios, f, indent=4)
        messagebox.showinfo("シナリオ保存", f"シナリオを保存しました: {filename}")

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            self.scenarios = json.load(f)

class AnalyticsDashboard:
    def __init__(self, history):
        self.history = history

    def plot_metrics(self):
        if not self.history:
            messagebox.showerror("データ分析", "履歴データがありません。")
            return
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        axs[0].plot([h['speed'] for h in self.history], label='Speed (m/s)')
        axs[0].set_title('Speed Profile')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Speed (m/s)')
        axs[0].legend()
        
        axs[1].plot([math.degrees(h['steering']) for h in self.history], label='Steering Angle (°)', color='orange')
        axs[1].set_title('Steering Angle')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Steering Angle (°)')
        axs[1].legend()
        
        axs[2].plot([h['cte'] for h in self.history], label='Cross Track Error (m)', color='red')
        axs[2].set_title('Cross Track Error')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('CTE (m)')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
