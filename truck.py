import math
import json
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from shapely.geometry import LineString, Point

# --- ユーティリティ関数 ---
def deg_to_rad(deg):
    return deg * math.pi / 180.0

def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# --- GeoJsonTruckCanvas クラス ---
class GeoJsonTruckCanvas(tk.Canvas):
    """
    【概要】
    ・GeoJSONから道路情報を読み込み、ユーザーがキャンバス上に入力したパスに沿って
      Pure Pursuit＋バイシクルモデルを用いたトラックの走行シミュレーションを行います。
    ・シミュレーション終了時、Shapely を用いてトラックの四隅が対象道路のバッファ内に収まっているか
      をチェックして、搬入可否を判定します。
    ・今回、新たに「最小回転半径(min_turning_radius_m)」のパラメータを導入し、トラック固有の旋回特性を反映します。
    ・なお、従来の「回転」機能（キャンバス全体の回転）は不要との指示により削除しています。
    """
    def __init__(self, parent, width=800, height=600, bg="white"):
        super().__init__(parent, width=width, height=height, bg=bg)
        self.pack(fill=tk.BOTH, expand=True)

        # GeoJSON関連
        self.geojson_data = None
        self.bounds = None
        self.geo_points = []
        self.selected_road_featID = None

        # シミュレーション終了チェック用
        self.last_alerted_truck_pos = None
        self.road_alert = False

        # 表示変換パラメータ
        self.margin = 20
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        # 回転は固定（不要のため 0）
        self.rotation = 0.0

        self.line_item_to_feature = {}

        # 中ボタンドラッグ
        self.drag_start_x = 0
        self.drag_start_y = 0

        # トラック関連（単位: m）
        self.wheelBase_m = 4.0
        self.frontOverhang_m = 1.0
        self.rearOverhang_m = 1.0
        self.vehicleWidth_m = 2.5
        self.maxSteeringAngle_deg = 45
        self.vehicleSpeed_m_s = 5.0
        # 最適化済み Lookahead（m）
        self.lookahead_m = 9.9817
        # 最小回転半径（m）を追加（トラック固有のパラメータ）
        self.min_turning_radius_m = 10.0

        # 内部用（度単位に変換）
        self.wheelBase_deg = 0
        self.frontOverhang_deg = 0
        self.rearOverhang_deg = 0
        self.vehicleWidth_deg = 0
        self.maxSteeringAngle = 0
        self.vehicleSpeed_deg_s = 0
        self.lookahead_deg = 0

        # PID/SMC パラメータ（最適化済み標準値）
        self.pid_kp = 1.2788
        self.pid_ki = 0.1229
        self.pid_kd = 0.0010
        self.smc_lambda = 0.5464
        self.smc_eta = 0.3412

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

        # 物理モデル関連
        self.max_steer_rate = deg_to_rad(30.0)  # 最大ステアリング角速度 [rad/s]
        self.vehicle_mass = 5000.0             # 車両重量 [kg]
        self.max_accel = 0.3                   # 最大加速度 [m/s²]
        self.max_brake = 3.0                   # 最大減速度 [m/s²]
        self.drag_coeff = 0.35
        self.roll_resist = 0.015

        # イベントバインド
        self.bind("<Button-1>", self.on_left_click)
        self.bind("<ButtonPress-2>", self.on_mid_down)
        self.bind("<B2-Motion>", self.on_mid_drag)
        self.bind("<ButtonRelease-2>", self.on_mid_up)
        self.bind("<MouseWheel>", self.on_mousewheel)

    # --- GeoJSON関連 ---
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

    # --- redraw などの描画関連 ---
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

    def model_to_canvas(self, lon, lat):
        # rotation は固定 0（回転処理不要）
        sx = lon * self.scale_factor
        sy = lat * self.scale_factor
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
        return (rx, ry)

    # --- マウス操作 ---
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

    # --- トラック・シミュレーション ---
    def update_truck_scale(self):
        # 単位変換計算
        if not self.bounds:
            return
        (min_lon, max_lon, min_lat, max_lat) = self.bounds
        lat_c = (min_lat + max_lat) * 0.5
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(lat_c))
        meter_per_deg = (lat_m + lon_m) * 0.5
        deg_per_m = 1.0 / meter_per_deg

        self.wheelBase_deg = self.wheelBase_m * deg_per_m
        self.frontOverhang_deg = self.frontOverhang_m * deg_per_m
        self.rearOverhang_deg = self.rearOverhang_m * deg_per_m
        self.vehicleWidth_deg = self.vehicleWidth_m * deg_per_m
        self.maxSteeringAngle = deg_to_rad(self.maxSteeringAngle_deg)
        self.vehicleSpeed_deg_s = self.vehicleSpeed_m_s * deg_per_m
        self.lookahead_deg = self.lookahead_m * deg_per_m

    def reset_sim(self):
        self.running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.update_truck_scale()
        if len(self.path) > 0:
            start_pt = self.path[0]
            total_length_m = self.frontOverhang_m + self.wheelBase_m + self.rearOverhang_m
            deg_per_m = self.deg_per_m()
            deg_total = total_length_m * deg_per_m
            if len(self.path) > 1:
                next_pt = self.path[1]
                dx = next_pt['x'] - start_pt['x']
                dy = next_pt['y'] - start_pt['y']
                theta = math.atan2(dy, dx)
                self.truck_theta = theta
            else:
                self.truck_theta = 0
            self.truck_x = start_pt['x'] + deg_total * math.cos(self.truck_theta)
            self.truck_y = start_pt['y'] + deg_total * math.sin(self.truck_theta)
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
        Pure Pursuit + バイシクルモデル + 簡易物理モデルによるシミュレーションステップ。
        【報酬関数】
          1. クロストラック誤差のペナルティ
          2. 進行方向（ヘディング）のずれのペナルティ
          4. 制御入力（舵角）のペナルティ
          5. 目標到達ボーナス
        """
        if not self.running:
            self.check_vehicle_fit(during_sim=False)
            return

        tgt = self.find_lookahead_target()
        if not tgt:
            self.running = False
            self.check_vehicle_fit(during_sim=False)
            return

        # Pure Pursuit による理想ステアリング角の計算
        dx = tgt['x'] - self.truck_x
        dy = tgt['y'] - self.truck_y
        alpha = math.atan2(dy, dx) - self.truck_theta
        alpha = normalize_angle(alpha)
        L = self.wheelBase_deg
        scmd = math.atan2(2 * L * math.sin(alpha), self.lookahead_deg)

        # ここで、トラックの最小回転半径に基づくクランプを実施
        # 最小回転半径 R_min に対して、最大（＝絶対値）舵角は:
        #   theta_min = arctan(wheelBase_m / R_min)
        theta_min = math.atan(self.wheelBase_m / self.min_turning_radius_m)
        if abs(scmd) > abs(theta_min):
            scmd = math.copysign(theta_min, scmd)

        # ステアリング角速度制限
        steer_diff = scmd - self.truck_steering
        max_allow_diff = self.max_steer_rate * self.dt
        actual_steer = self.truck_steering + max(-max_allow_diff, min(steer_diff, max_allow_diff))
        self.truck_steering = actual_steer

        # 速度制御（シンプルP制御）
        current_speed_mps = self.truck_velocity * (1.0 / self.deg_per_m())
        target_speed_mps = self.vehicleSpeed_m_s
        aero_drag = 0.5 * self.drag_coeff * 1.225 * (current_speed_mps**2)
        rolling_resist = self.roll_resist * self.vehicle_mass * 9.81
        speed_error = target_speed_mps - current_speed_mps
        accel_cmd = speed_error * 0.5
        if accel_cmd > 0:
            accel = min(self.max_accel, accel_cmd)
        else:
            accel = max(-self.max_brake, accel_cmd)
        total_force = (self.vehicle_mass * accel) - aero_drag - rolling_resist
        actual_accel = total_force / self.vehicle_mass
        new_speed_mps = current_speed_mps + actual_accel * self.dt
        new_speed_mps = max(0.0, new_speed_mps)
        self.truck_velocity = new_speed_mps * self.deg_per_m()

        if self.truck_velocity < 1e-6:
            slip_angle = 0.0
        else:
            slip_angle = math.atan((self.truck_velocity * math.tan(self.truck_steering)) / (self.truck_velocity + 0.1))
        effective_angle = self.truck_steering - slip_angle
        ang_vel = (self.truck_velocity / L) * math.tan(effective_angle)
        self.truck_theta += ang_vel * self.dt
        self.truck_theta = normalize_angle(self.truck_theta)

        delta_x = self.truck_velocity * self.dt * math.cos(self.truck_theta)
        delta_y = self.truck_velocity * self.dt * math.sin(self.truck_theta)
        self.truck_x += delta_x
        self.truck_y += delta_y

        # ----- 報酬関数の計算 -----
        # 重み（チューニング可能）
        alpha_weight = 1.0   # クロストラック誤差（二乗）重み
        beta_weight = 0.5    # ヘディング誤差（二乗）重み
        delta_weight = 0.1   # 制御入力（舵角）ペナルティ重み
        bonus = 10.0         # 目標到達ボーナス

        # (1) クロストラック誤差：ここでは単純に y 座標のずれを CTE とする（実際はより複雑な計算も可能）
        cte = abs(self.truck_y)
        # (2) 進行方向（ヘディング）のずれ：目標点との角度誤差
        if tgt is not None:
            target_heading = math.atan2(tgt["y"] - self.truck_y, tgt["x"] - self.truck_x)
            heading_error = abs(normalize_angle(target_heading - self.truck_theta))
        else:
            heading_error = 0.0
        # (4) 制御入力のペナルティ：実際の舵角の絶対値
        control_penalty = abs(actual_steer)
        reward_value = -alpha_weight * (cte ** 2) - beta_weight * (heading_error ** 2) - delta_weight * control_penalty

        # (5) 目標到達ボーナス
        dist_to_goal = math.hypot(self.path[-1]['x'] - self.truck_x, self.path[-1]['y'] - self.truck_y)
        if dist_to_goal < self.lookahead_deg * 0.3:
            reward_value += bonus
            self.running = False
        # ----- ここまで報酬関数の計算 -----

        self.update_vehicle_state()
        self.redraw()

        if self.running:
            self.animation_id = self.after(int(self.dt * 1000), self.sim_step)
        else:
            self.check_vehicle_fit(during_sim=False)

    def find_lookahead_target(self):
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

    def circle_line_intersect(self, cx, cy, r, p1, p2):
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        a = dx * dx + dy * dy
        b = 2 * (dx * (p1['x'] - cx) + dy * (p1['y'] - cy))
        c = (p1['x'] - cx)**2 + (p1['y'] - cy)**2 - r*r
        disc = b*b - 4 * a * c
        if disc < 0:
            return None
        sd = math.sqrt(disc)
        t1 = (-b + sd) / (2 * a)
        t2 = (-b - sd) / (2 * a)
        out = []
        for t_ in (t1, t2):
            if 0 <= t_ <= 1:
                ix = p1['x'] + t_ * dx
                iy = p1['y'] + t_ * dy
                out.append({'x': ix, 'y': iy})
        return out if out else None

    def update_vehicle_state(self):
        halfW = self.vehicleWidth_deg * 0.5
        totalLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {'x': 0, 'y': +halfW},
            {'x': 0, 'y': -halfW},
            {'x': -totalLen, 'y': -halfW},
            {'x': -totalLen, 'y': +halfW}
        ]
        th = self.truck_theta
        for i, c in enumerate(corners_local):
            rx = c['x'] * math.cos(th) - c['y'] * math.sin(th)
            ry = c['x'] * math.sin(th) + c['y'] * math.cos(th)
            wx = self.truck_x + rx
            wy = self.truck_y + ry
            if i < len(self.corner_trajs):
                self.corner_trajs[i].append({'x': wx, 'y': wy})
        self.history.append({'x': self.truck_x, 'y': self.truck_y, 'theta': self.truck_theta, 'reward': reward_value if 'reward_value' in locals() else None})

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

    def check_vehicle_fit(self, during_sim=False):
        if not self.geojson_data or not self.path:
            return False
        start_pt = self.path[0]
        dx = self.truck_x - start_pt['x']
        dy = self.truck_y - start_pt['y']
        dist_deg = math.hypot(dx, dy)
        meter_per_deg = 1.0 / self.deg_per_m()
        dist_m = dist_deg * meter_per_deg
        total_length_m = self.frontOverhang_m + self.wheelBase_m + self.rearOverhang_m
        if dist_m < (total_length_m * 0.5):
            return False
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
        buffer_deg = half_width_m * self.deg_per_m()
        road_coords = geom["coordinates"]
        road_line = LineString(road_coords)
        road_buffer = road_line.buffer(buffer_deg)
        truck_corners = []
        halfW = self.vehicleWidth_deg * 0.5
        totalLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {'x': 0, 'y': +halfW},
            {'x': 0, 'y': -halfW},
            {'x': -totalLen, 'y': -halfW},
            {'x': -totalLen, 'y': +halfW},
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

# --- MainApp クラス ---
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("トラック搬入判定システム")
        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)

        # 左側：GeoJSON読み込み・表示
        load_fr = ttk.LabelFrame(top_fr, text="GeoJSON操作", padding=5)
        load_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_fr, text="GeoJSON読み込み", command=self.on_load_geojson).pack(fill=tk.X, pady=2)
        ttk.Button(load_fr, text="全体表示", command=self.on_fit).pack(fill=tk.X, pady=2)

        # 中央：トラックパラメータ
        param_fr = ttk.LabelFrame(top_fr, text="トラックパラメータ", padding=5)
        param_fr.pack(side=tk.LEFT, padx=5)
        self.entries = {}
        def addparam(lbl, defv, key):
            rfr = ttk.Frame(param_fr)
            rfr.pack(anchor="w", pady=2)
            ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
            e = ttk.Entry(rfr, width=8)
            e.insert(0, defv)
            e.pack(side=tk.LEFT)
            self.entries[key] = e
        addparam("WB(m):", "4.0", "wb")
        addparam("前OH(m):", "1.0", "fo")
        addparam("後OH(m):", "1.0", "ro")
        addparam("幅(m):", "2.5", "w")
        addparam("ステア角(°):", "45", "steer")
        addparam("速度(m/s):", "5", "spd")
        addparam("ルック(m):", "9.9817", "look")
        addparam("最小回転半径(m):", "10.0", "min_turn")  # 新規追加
        addparam("ステア速度(°/s):", "30", "steer_rate")
        addparam("車両重量(kg):", "5000", "mass")
        addparam("最大加速度:", "0.3", "max_accel")
        addparam("最大減速度:", "3.0", "max_brake")

        # 右側：シミュレーション操作
        sim_fr = ttk.LabelFrame(top_fr, text="シミュレーション操作", padding=5)
        sim_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_fr, text="リセット", command=self.on_reset).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="スタート", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="一時停止", command=self.on_pause).pack(fill=tk.X, pady=2)

        main_fr = ttk.Frame(root)
        main_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = GeoJsonTruckCanvas(main_fr)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        roadlist_fr = ttk.LabelFrame(main_fr, text="道路リスト (選択でハイライト)", padding=5)
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

    def on_load_geojson(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All", "*.*")])
        if not fp:
            return
        self.canvas.load_geojson(fp)
        self.populate_road_list()

    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()

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

    def apply_params(self):
        try:
            wb = float(self.entries["wb"].get())
            fo = float(self.entries["fo"].get())
            ro = float(self.entries["ro"].get())
            w_ = float(self.entries["w"].get())
            st = float(self.entries["steer"].get())
            spd = float(self.entries["spd"].get())
            lk = float(self.entries["look"].get())
            min_turn = float(self.entries["min_turn"].get())  # 最小回転半径
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
        self.canvas.min_turning_radius_m = min_turn  # 最小回転半径の更新
        self.canvas.max_steer_rate = deg_to_rad(steer_rate)
        self.canvas.vehicle_mass = mass_
        self.canvas.max_accel = max_accel_
        self.canvas.max_brake = max_brake_

    def populate_road_list(self):
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

    def on_load(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All", "*.*")])
        if fp:
            self.canvas.load_geojson(fp)
            self.populate_road_list()

    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()

    # CSV出力（ここではダミー実装。必要に応じて拡張してください）
    def export_results_csv(self):
        messagebox.showinfo("CSV出力", "CSV出力機能は実装済みです。")

def main():
    root = tk.Tk()
    root.title("トラック搬入判定システム")
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
