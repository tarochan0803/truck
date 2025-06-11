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
    PurePursuit + バイシクルモデル + PID + SMC
    + 各種改良(動的ルックアヘッド、正確なPacejka、適応型PID、KDTree評価)
    
    【主要な改善点】
    - KDTreeの再構築はGeoJSON読込時のみ (パフォーマンス向上)
    - バッチモード実行の安定性 (実行中判定)
    - 終了閾値を lookahead の 0.1倍に変更 (より正確に最終点を通過するまで走行)
    - パラメータ最適化中にシミュレーションを走らせない工夫 (batch_modeフラグ操作)
    - 各種パラメータ範囲を現実的に
    - シミュレーションを速くするため dt を小さく、after間隔も短く
    """
    def __init__(self, parent, width=800, height=600, bg="white"):
        super().__init__(parent, width=width, height=height, bg=bg)
        self.pack(fill=tk.BOTH, expand=True)

        # GeoJSON
        self.geojson_data = None
        self.bounds = None
        self.geo_points = []
        self.selected_road_featID = None

        # 表示用
        self.margin = 20
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.rotation = 0.0

        # (将来的にアイテムとフィーチャIDを紐付けたい場合のメモ)
        self.line_item_to_feature = {}

        # 中ボタンドラッグ
        self.drag_start_x = 0
        self.drag_start_y = 0

        # 車両パラメータ
        self.wheelBase_m = 4.0
        self.frontOverhang_m = 1.0
        self.rearOverhang_m = 1.0
        self.vehicleWidth_m = 2.5
        self.maxSteeringAngle_deg = 45
        self.vehicleSpeed_m_s = 5.0
        self.lookahead_m = 10.0

        # バイシクルモデル用度単位
        self.wheelBase_deg = 0
        self.frontOverhang_deg = 0
        self.rearOverhang_deg = 0
        self.vehicleWidth_deg = 0
        self.maxSteeringAngle = 0
        self.vehicleSpeed_deg_s = 0
        self.lookahead_deg = 0

        # 物理モデル
        self.max_steer_rate = deg_to_rad(30)  # 1ステップで変化可能な最大舵角速度
        self.vehicle_mass = 5000
        self.max_accel = 0.3
        self.max_brake = 3.0
        self.drag_coeff = 0.35
        self.roll_resist = 0.015

        # PID/SMC
        self.pid_kp = 0.8
        self.pid_ki = 0.01
        self.pid_kd = 0.05
        self.error_integral = 0.0
        self.last_error = 0.0

        self.smc_lambda = 0.5
        self.smc_eta = 0.2

        # 速度PID
        self.speed_kp = 1.0
        self.speed_ki = 0.1
        self.speed_kd = 0.05
        self.speed_integral = 0.0
        self.last_speed_error = 0.0

        # シミュ状態
        self.path = []
        self.running = False
        self.animation_id = None

        # シミュレーションを高速化するため、dt小さめ(例:0.05)に変更
        self.dt = 0.05

        # 状態
        self.truck_x = 0
        self.truck_y = 0
        self.truck_theta = 0
        self.truck_velocity = 0
        self.truck_steering = 0

        # ログ
        self.corner_trajs = [[], [], [], []]
        self.history = []
        self.alert_marker = None

        # KDTree for road (初期はNone、ロード後にbuild_road_treeで構築)
        self.road_tree = None

        # シミュ終了後に (True=はみ出し, False=収まる)
        self.corners_outside = None

        # イベント
        self.bind("<Button-1>", self.on_left_click)
        self.bind("<ButtonPress-2>", self.on_mid_down)
        self.bind("<B2-Motion>", self.on_mid_drag)
        self.bind("<ButtonRelease-2>", self.on_mid_up)
        self.bind("<MouseWheel>", self.on_mousewheel)

    # ----------- GeoJSON関連 -----------
    def load_geojson(self, fp):
        with open(fp, "r", encoding="utf-8") as f:
            self.geojson_data = json.load(f)
        self.extract_geo_points()
        self.compute_bounds()
        self.full_view()
        # ここでKDTreeをあらかじめ構築しておく (パフォーマンス改善)
        self.build_road_tree()
        self.redraw()

    def build_road_tree(self):
        """roadのLineString座標をまとめてKDTree化"""
        self.road_tree = None
        if not self.geojson_data:
            return
        feats = self.geojson_data.get("features", [])
        road_points = []
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                road_points.extend(coords)
        if road_points:
            self.road_tree = KDTree(road_points)

    def extract_geo_points(self):
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
        if vw < 10 or vh < 10:
            return
        scx = vw / dx
        scy = vh / dy
        s = min(scx, scy)
        cx, cy = w * 0.5, h * 0.5
        midx = (minx + maxx) * 0.5
        midy = (miny + maxy) * 0.5
        self.scale_factor = s
        self.offset_x = cx - s * midx
        self.offset_y = cy + s * midy

    # ----------- 描画関連 -----------
    def redraw(self):
        self.delete("all")

        # 道路(LineString)
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

        # パス(青)
        if len(self.path) > 1:
            arr = []
            for p_ in self.path:
                px, py = self.model_to_canvas(p_['x'], p_['y'])
                arr.extend([px, py])
            self.create_line(*arr, fill="blue", width=2)

        # corner_trajs
        colz = ["red", "green", "blue", "orange"]
        for i, traj in enumerate(self.corner_trajs):
            if len(traj) < 2:
                continue
            arr = []
            for wpt in traj:
                sx, sy = self.model_to_canvas(wpt["x"], wpt["y"])
                arr.extend([sx, sy])
            self.create_line(*arr, fill=colz[i], dash=(2, 2), width=1)

        # トラック
        if self.history:
            self.draw_truck()

    def deg_per_m(self):
        if not self.bounds:
            return 1e-6
        (minx, maxx, miny, maxy) = self.bounds
        lat_c = (miny + maxy) * 0.5
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(lat_c))
        meter_per_deg = (lat_m + lon_m) * 0.5
        return 1.0 / meter_per_deg

    def deg_to_px(self, dval):
        x1, y1 = self.model_to_canvas(0, 0)
        x2, y2 = self.model_to_canvas(dval, 0)
        return max(1, int(math.hypot(x2 - x1, y2 - y1)))

    def model_to_canvas(self, x, y):
        # 回転を適用
        rx = x * math.cos(self.rotation) - y * math.sin(self.rotation)
        ry = x * math.sin(self.rotation) + y * math.cos(self.rotation)
        # スケーリング
        sx = rx * self.scale_factor
        sy = ry * self.scale_factor
        # オフセット & y上下反転
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

    # ----------- マウス操作 -----------
    def on_left_click(self, e):
        mx, my = e.x, e.y
        lon, lat = self.canvas_to_model(mx, my)
        self.path.append({"x": lon, "y": lat})
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

    # ----------- シミュレーション関連 -----------
    def update_truck_scale(self):
        dpm = self.deg_per_m()
        self.wheelBase_deg = self.wheelBase_m * dpm
        self.frontOverhang_deg = self.frontOverhang_m * dpm
        self.rearOverhang_deg = self.rearOverhang_m * dpm
        self.vehicleWidth_deg = self.vehicleWidth_m * dpm
        self.maxSteeringAngle = deg_to_rad(self.maxSteeringAngle_deg)
        self.vehicleSpeed_deg_s = self.vehicleSpeed_m_s * dpm
        self.lookahead_deg = self.lookahead_m * dpm

    def reset_sim(self):
        self.running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.update_truck_scale()

        if self.path:
            # パスの先頭を基準に後退配置
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
        self.corners_outside = None

        self.error_integral = 0.0
        self.last_error = 0.0
        self.speed_integral = 0.0
        self.last_speed_error = 0.0

        self.redraw()

    def start_sim(self):
        if len(self.path) < 1:
            messagebox.showerror("エラー", "パスが足りません。")
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
        (1) 動的ルックアヘッド
        (2) スリップ角補償
        (3) 適応型PID + SMC
        (4) 速度PID
        (5) 逸脱検知(KDTree)
        - 終了条件は「最終パス点に十分近づいたらのみ」
        """
        if not self.running:
            self.corners_outside = self.check_vehicle_fit(during_sim=False, no_popup=True)
            self.event_generate("<<SimFinished>>", when="tail")
            return

        # 1. 動的ルックアヘッド距離
        speed_mps = self.truck_velocity * (1.0 / self.deg_per_m())
        self.lookahead_m = max(3.0, min(15.0, speed_mps * 1.2))
        self.lookahead_deg = self.lookahead_m * self.deg_per_m()

        # 目標点探索
        tgt = self.find_lookahead_target()

        if tgt is None:
            # 見つからなくても終了しない。ステアリングは0にして進む例
            combined_steer = 0.0
        else:
            # CTE計算
            closest_pt = self.find_closest_path_point()
            dx = closest_pt["x"] - self.truck_x
            dy = closest_pt["y"] - self.truck_y
            cte = math.hypot(dx, dy) * math.sin(math.atan2(dy, dx) - self.truck_theta)
            self.cross_track_error = cte

            # 適応型PID
            cte_mag = abs(cte)
            self.pid_kp = 1.2 / (1 + 0.5 * cte_mag)   # 簡易的な非線形ゲイン
            self.pid_kd = 0.1 * (1 + cte_mag)

            if cte_mag < 0.5:
                self.error_integral += cte * self.dt
            else:
                self.error_integral *= 0.95

            d_cte = (cte - self.last_error) / self.dt
            pid_out = self.pid_kp * cte + self.pid_ki * self.error_integral + self.pid_kd * d_cte
            self.last_error = cte

            # SMC
            sliding_surf = d_cte + self.smc_lambda * cte
            if abs(sliding_surf) > 1e-3:
                smc_out = -self.smc_eta * math.copysign(1, sliding_surf)
            else:
                smc_out = 0

            # PurePursuit
            dx2 = tgt["x"] - self.truck_x
            dy2 = tgt["y"] - self.truck_y
            alpha = math.atan2(dy2, dx2) - self.truck_theta
            alpha = normalize_angle(alpha)
            L = self.wheelBase_deg
            scmd = math.atan2(2.0 * L * math.sin(alpha), self.lookahead_deg)
            scmd = max(-self.maxSteeringAngle, min(self.maxSteeringAngle, scmd))

            combined_steer = scmd + pid_out + smc_out
            combined_steer = max(-self.maxSteeringAngle, min(self.maxSteeringAngle, combined_steer))

        # ステア速度制限
        steer_diff = combined_steer - self.truck_steering
        max_diff = self.max_steer_rate * self.dt
        actual_steer = self.truck_steering + max(-max_diff, min(steer_diff, max_diff))
        self.truck_steering = actual_steer

        # 速度PID
        current_speed_mps = self.truck_velocity / (self.deg_per_m())
        speed_err = self.vehicleSpeed_m_s - current_speed_mps
        self.speed_error = speed_err

        if abs(speed_err) < 0.5:
            self.speed_integral += speed_err * self.dt
        else:
            self.speed_integral = 0

        d_spd = (speed_err - self.last_speed_error) / self.dt
        accel_cmd = (self.speed_kp * speed_err
                     + self.speed_ki * self.speed_integral
                     + self.speed_kd * d_spd)
        self.last_speed_error = speed_err

        if accel_cmd > 0:
            accel = min(self.max_accel, accel_cmd)
        else:
            accel = max(-self.max_brake, accel_cmd)

        # 抵抗
        aero_drag = 0.5 * self.drag_coeff * 1.225 * (current_speed_mps ** 2)
        rolling = self.roll_resist * self.vehicle_mass * 9.81
        total_force = (self.vehicle_mass * accel) - aero_drag - rolling
        actual_accel = total_force / self.vehicle_mass

        # 2. スリップ角補償 (Pacejka)
        slip_angle = math.atan2(self.wheelBase_m * math.tan(self.truck_steering),
                                self.wheelBase_m)
        B, C, D, E = 10.0, 1.9, 1.0, 0.97
        pacejka_out = D * math.sin(
            C * math.atan(B * slip_angle - E * (B * slip_angle - math.atan(B * slip_angle)))
        )
        Fy = pacejka_out * 1000
        self.acceleration = (Fy - rolling) / self.vehicle_mass
        actual_accel += self.acceleration

        new_speed_mps = current_speed_mps + actual_accel * self.dt
        if new_speed_mps < 0:
            new_speed_mps = 0
        self.truck_velocity = new_speed_mps * self.deg_per_m()

        # スリップ
        if self.truck_velocity < 1e-6:
            slip = 0
        else:
            slip = math.atan(
                (self.truck_velocity * math.tan(self.truck_steering))
                / (self.truck_velocity + 0.1)
            )
        eff_steer = self.truck_steering - slip

        # バイシクルモデル更新
        L = self.wheelBase_deg
        ang_v = (self.truck_velocity / L) * math.tan(eff_steer) if abs(L) > 1e-9 else 0
        self.truck_theta += ang_v * self.dt
        self.truck_theta = normalize_angle(self.truck_theta)

        vx = self.truck_velocity * self.dt * math.cos(self.truck_theta)
        vy = self.truck_velocity * self.dt * math.sin(self.truck_theta)
        self.truck_x += vx
        self.truck_y += vy

        # 状態更新
        self.update_vehicle_state()
        # 不要な再描画を減らす場合は条件により呼ばないようにできるが、ここでは都度描画
        self.redraw()

        # 5. 逸脱検知(警告表示のみ止めない)
        out = self.check_vehicle_fit(during_sim=True, no_popup=True)
        if out:
            self.create_vehicle_alert_marker()

        # 終了判定: 最終パス点に十分近づいたら終了
        dist_fin = math.hypot(
            self.path[-1]["x"] - self.truck_x,
            self.path[-1]["y"] - self.truck_y
        )
        # ★閾値を緩和 (0.3→0.1に変更)
        if dist_fin < self.lookahead_deg * 0.1:
            self.running = False

        if self.running:
            # シミュレーションを速めるためにafter時間を短くする (例: int(self.dt*1000*0.5))
            self.animation_id = self.after(int(self.dt * 1000 * 0.5), self.sim_step)
        else:
            self.corners_outside = self.check_vehicle_fit(during_sim=False, no_popup=True)
            self.event_generate("<<SimFinished>>", when="tail")

    def find_lookahead_target(self):
        cx, cy = self.truck_x, self.truck_y
        L = self.lookahead_deg
        best = None
        best_d = 999999
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
            # パス末端しかなければ末端を返す
            best = self.path[-1]
        return best

    def find_closest_path_point(self):
        best = None
        bestd = 999999
        for pt in self.path:
            dd = math.hypot(pt["x"] - self.truck_x, pt["y"] - self.truck_y)
            if dd < bestd:
                bestd = dd
                best = pt
        return best if best else self.path[-1]

    def circle_line_intersect(self, cx, cy, r, p1, p2):
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

    def check_vehicle_fit(self, during_sim=False, no_popup=False):
        """
        True => はみ出し(衝突想定), False => 収まる
        """
        if not self.geojson_data or not self.path:
            return False

        # 既にbuild_road_tree済みなので再構築しない
        if self.road_tree is None:
            return False

        # corner_trajsの最後の位置（今ステップ更新直後）で判定
        min_distances = []
        for i in range(4):
            if self.corner_trajs[i]:
                last_corner = self.corner_trajs[i][-1]
                dist, idx = self.road_tree.query([(last_corner["x"], last_corner["y"])])
                min_distances.append(dist[0])

        if not min_distances:
            return False

        vehicle_width_deg = self.vehicleWidth_m * self.deg_per_m()
        threshold = vehicle_width_deg * 1.2
        return any(d > threshold for d in min_distances)

    def update_vehicle_state(self):
        halfW = self.vehicleWidth_deg * 0.5
        totLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {"x": 0,       "y": +halfW},
            {"x": 0,       "y": -halfW},
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

        cte_val = 0
        if hasattr(self, "cross_track_error"):
            cte_val = self.cross_track_error
        self.history.append({
            "x": self.truck_x,
            "y": self.truck_y,
            "theta": self.truck_theta,
            "speed": self.truck_velocity * (1.0 / self.deg_per_m()),
            "steering": self.truck_steering,
            "cte": cte_val,
        })

    def draw_truck(self):
        if not self.history:
            return
        last = self.history[-1]
        tx, ty = last["x"], last["y"]
        th = last["theta"]
        halfW = self.vehicleWidth_deg * 0.5
        totLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {"x": 0,       "y": +halfW},
            {"x": 0,       "y": -halfW},
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
        if self.alert_marker:
            self.delete(self.alert_marker)
            self.alert_marker = None
        px, py = self.model_to_canvas(self.truck_x, self.truck_y)
        self.alert_marker = self.create_oval(px - 15, py - 15, px + 15, py + 15,
                                             outline="red", width=3)
        self.after(1000, self.delete_alert_marker)

    def delete_alert_marker(self):
        if self.alert_marker is not None:
            self.delete(self.alert_marker)
            self.alert_marker = None


class MainApp:
    """
    連続モードでシミュを繰り返し + 最適化
    + 可視化拡張(四隅軌跡 / パラメータ最適化)
    
    【主要な改善点】
    - 連続実行の安定性向上:
      * run_batch_stepで前のシミュが終了するまで待機
      * シミュ終了後に少し遅延を入れて次を実行
    - 最適化中はバッチモードを止めてシミュも停止
    - ランダムパスは20点とする
    """
    def __init__(self, root):
        self.root = root
        self.root.title("PurePursuit + PID + SMC + 改良 & 連続モード (省略なし)")

        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_fr, text="GeoJSON読み込み", command=self.on_load_geojson).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="連続開始", command=self.start_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="連続停止", command=self.stop_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="結果を見る", command=self.show_results_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_fr, text="パラメータ最適化", command=self.optimize_parameters).pack(side=tk.LEFT, padx=5)

        self.canvas = GeoJsonTruckCanvas(root)
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # バッチモード管理
        self.batch_mode = False
        self.batch_results = []

        self.canvas.bind("<<SimFinished>>", self.on_sim_finished)

    # ---------------------- ボタンコールバック ----------------------
    def on_load_geojson(self):
        fp = filedialog.askopenfilename(
            filetypes=[("GeoJSON", "*.geojson"), ("All", "*.*")]
        )
        if not fp:
            return
        self.canvas.load_geojson(fp)

    def start_batch(self):
        if not self.canvas.geojson_data:
            messagebox.showerror("GeoJSONなし", "先にGeoJSONを読み込んでください。")
            return
        self.batch_mode = True
        self.batch_results.clear()
        self.run_batch_step()

    def stop_batch(self):
        self.batch_mode = False
        self.canvas.pause_sim()
        messagebox.showinfo("停止", "連続モードを終了しました。")

    def run_batch_step(self):
        """
        ランダムroadWidth, 車両パラメータ, パス -> reset -> start
        前回のシミュがまだrunningなら100ms後にリトライ
        """
        if not self.batch_mode:
            return  # バッチモードでなければ実行しない

        # 前のシミュがまだ走り終わっていない場合、少し待ってリトライ
        if self.canvas.running:
            self.root.after(100, self.run_batch_step)
            return

        self.randomize_road_widths()
        self.randomize_vehicle_params()
        # ランダムパスは20点とする
        self.pick_random_path(npts=20)

        self.canvas.reset_sim()
        self.canvas.start_sim()

    def on_sim_finished(self, event):
        now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feats = self.canvas.geojson_data.get("features", [])
        roads = []
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                props = ft.get("properties", {})
                roads.append(props.get("roadWidthM", None))

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
                "speed_kd": self.canvas.speed_kd,
            },
            "road_widths": roads,
            "corners_outside": self.canvas.corners_outside,
            "path": self.canvas.path[:],
            "history": self.canvas.history[:]
        }
        self.batch_results.append(scenario)
        print(f"[SimFinished] time={now_s}, corners_out={self.canvas.corners_outside}, hist_len={len(self.canvas.history)}")

        # バッチモード中なら少し遅延して次を実行
        if self.batch_mode:
            self.root.after(100, self.run_batch_step)

    def show_results_list(self):
        if not self.batch_results:
            messagebox.showinfo("結果なし", "まだ実行していません。")
            return

        win = tk.Toplevel(self.root)
        win.title("連続実行結果一覧")

        tree = ttk.Treeview(win, columns=("time", "out"), show="headings", height=10)
        tree.heading("time", text="Time")
        tree.heading("out", text="CornersOut")
        tree.column("time", width=150)
        tree.column("out", width=80)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        for i, scen in enumerate(self.batch_results):
            tstr = scen["time"]
            out = scen["corners_outside"]
            tree.insert("", tk.END, values=(tstr, out))

        def on_dbl(e):
            sel = tree.selection()
            if not sel:
                return
            iid = sel[0]
            idx = tree.index(iid)
            scenario = self.batch_results[idx]
            self.show_scenario_detail(scenario)

        tree.bind("<Double-1>", on_dbl)

    def show_scenario_detail(self, scenario):
        dlg = tk.Toplevel(self.root)
        dlg.title("シミュ詳細")

        frm = ttk.Frame(dlg, padding=5)
        frm.pack(fill=tk.BOTH, expand=True)

        lb = tk.Listbox(frm, width=60, height=10)
        lb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        vp = scenario["vehicle_params"]
        lb.insert(tk.END, f"Time: {scenario['time']}")
        lb.insert(tk.END, f"CornersOut: {scenario['corners_outside']}")
        lb.insert(tk.END, f"RoadWidths: {scenario['road_widths']}")
        lb.insert(tk.END, f"WheelBase: {vp['wheelBase_m']:.2f}, Speed: {vp['vehicleSpeed_m_s']:.2f}")
        lb.insert(tk.END, f"PID_kp/ki/kd= {vp['pid_kp']:.2f}/{vp['pid_ki']:.2f}/{vp['pid_kd']:.2f}")
        lb.insert(tk.END, f"SMC_lambda/eta= {vp['smc_lambda']:.2f}/{vp['smc_eta']:.2f}")
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

        def plot_corners():
            fig, ax = plt.subplots()
            hist = scenario["history"]
            if not hist:
                messagebox.showinfo("No Hist", "履歴がありません。")
                return

            ccolors = ["red", "green", "blue", "orange"]
            for i in range(4):
                cxarr = []
                cyarr = []
                for h in hist:
                    halfW = scenario["vehicle_params"]["vehicleWidth_m"] * self.canvas.deg_per_m() * 0.5
                    totLen = (
                        scenario["vehicle_params"]["frontOverhang_m"]
                        + scenario["vehicle_params"]["wheelBase_m"]
                        + scenario["vehicle_params"]["rearOverhang_m"]
                    ) * self.canvas.deg_per_m()

                    corners_local = [
                        {"x": 0,       "y": +halfW},
                        {"x": 0,       "y": -halfW},
                        {"x": -totLen, "y": -halfW},
                        {"x": -totLen, "y": +halfW}
                    ]
                    th = h["theta"]
                    c_ = corners_local[i]
                    rx = c_["x"] * math.cos(th) - c_["y"] * math.sin(th)
                    ry = c_["x"] * math.sin(th) + c_["y"] * math.cos(th)
                    wx = h["x"] + rx
                    wy = h["y"] + ry
                    cxarr.append(wx)
                    cyarr.append(wy)
                ax.plot(cxarr, cyarr, color=ccolors[i], linestyle='--', linewidth=1)

            hx = [h["x"] for h in hist]
            hy = [h["y"] for h in hist]
            ax.plot(hx, hy, "r-", label="Vehicle Center")

            px = [p["x"] for p in scenario["path"]]
            py = [p["y"] for p in scenario["path"]]
            ax.plot(px, py, "b--", label="Reference Path")

            ax.set_aspect("equal")
            ax.legend()
            plt.title("四隅の軌跡")
            plt.show()

        corner_btn = ttk.Button(frm, text="四隅軌跡を表示", command=plot_corners)
        corner_btn.pack(side=tk.TOP, pady=5)

    # ------------ ランダム設定 ------------
    def randomize_road_widths(self):
        feats = self.canvas.geojson_data.get("features", [])
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                props = ft.setdefault("properties", {})
                props["roadWidthM"] = random.uniform(1.5, 6.0)

    def randomize_vehicle_params(self):
        # 現実的な範囲に調整
        self.canvas.wheelBase_m = random.uniform(3.0, 6.0)
        self.canvas.frontOverhang_m = random.uniform(0.5, 2.0)
        self.canvas.rearOverhang_m = random.uniform(0.5, 2.0)
        self.canvas.vehicleWidth_m = random.uniform(2.0, 3.0)
        self.canvas.maxSteeringAngle_deg = random.uniform(30, 45)
        self.canvas.vehicleSpeed_m_s = random.uniform(2.0, 8.0)
        self.canvas.lookahead_m = random.uniform(5.0, 15.0)

        # PID/SMC
        self.canvas.pid_kp = random.uniform(0.5, 1.5)
        self.canvas.pid_ki = random.uniform(0.0, 0.2)
        self.canvas.pid_kd = random.uniform(0.0, 0.1)
        self.canvas.smc_lambda = random.uniform(0.3, 0.8)
        self.canvas.smc_eta = random.uniform(0.1, 0.5)

        # 速度PID
        self.canvas.speed_kp = random.uniform(0.5, 2.0)
        self.canvas.speed_ki = random.uniform(0.0, 0.2)
        self.canvas.speed_kd = random.uniform(0.0, 0.1)

    def pick_random_path(self, npts=20):
        """
        ランダムにLineStringを選び、その中の連続する npts 点を切り出す
        該当ラインがなければシンプルに疑似生成
        """
        feats = self.canvas.geojson_data.get("features", [])
        lines = []
        for ft in feats:
            geom = ft.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if len(coords) >= npts:
                    lines.append(coords)
        if not lines:
            # データに十分なラインがない場合、擬似的に作成
            arr = [(0, 0)]
            for _ in range(npts - 1):
                arr.append(
                    (
                        arr[-1][0] + random.uniform(0.001, 0.01),
                        arr[-1][1] + random.uniform(0.001, 0.01),
                    )
                )
            self.canvas.path = [{"x": p[0], "y": p[1]} for p in arr]
            return
        line = random.choice(lines)
        max_start = len(line) - npts
        st = random.randint(0, max_start)
        subset = line[st : st + npts]
        self.canvas.path = [{"x": p[0], "y": p[1]} for p in subset]

    # ----------- (6) パラメータ最適化 -----------
    def optimize_parameters(self):
        """
        scipy.optimize.minimize でパラメータ最適化を行うサンプル
        ここでは corners_outside の回数を最小化するように... (簡略版)
        """
        # 最適化前にバッチモードを無効化
        prev_batch_mode = self.batch_mode
        self.batch_mode = False

        # シミュレーションを一時停止
        self.canvas.pause_sim()

        def objective(params):
            # params: [kp, ki, kd, lambda, eta]
            self.canvas.pid_kp = params[0]
            self.canvas.pid_ki = params[1]
            self.canvas.pid_kd = params[2]
            self.canvas.smc_lambda = params[3]
            self.canvas.smc_eta = params[4]

            # 一度だけシミュ (本来は複数回平均化を推奨)
            self.batch_results.clear()
            self.run_batch_step()  # 1回シミュ実行

            if not self.batch_results:
                return 1.0
            val = 1.0 if self.batch_results[-1]["corners_outside"] else 0.0
            return val

        initial_guess = [0.8, 0.01, 0.05, 0.5, 0.2]
        bounds = [
            (0.5, 1.5),  # kp
            (0.0, 0.2),  # ki
            (0.0, 0.1),  # kd
            (0.3, 0.8),  # lambda
            (0.1, 0.5)   # eta
        ]

        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='Powell')
            best_params = result.x
            msg = "最適パラメータ:\n" + str(best_params)
            messagebox.showinfo("最適化結果", msg)
        except Exception as e:
            messagebox.showerror("最適化エラー", str(e))
        finally:
            # 元のバッチモード状態に復元
            self.batch_mode = prev_batch_mode

    # ------------------------------------

def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
