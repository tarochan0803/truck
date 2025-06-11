import math
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import time
from datetime import datetime
from uuid import uuid4
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import psutil
import torch

from shapely.geometry import LineString, Point, Polygon

###############################################
# IntegratedGeoJsonTruckCanvas
###############################################
class IntegratedGeoJsonTruckCanvas(tk.Canvas):
    def __init__(self, parent, width=800, height=600, bg="white"):
        super().__init__(parent, width=width, height=height, bg=bg)
        self.pack(fill=tk.BOTH, expand=True)
        # --- シミュレーション用パラメータ ---
        self.geojson_data = None
        self.bounds = None
        self.geo_points = []
        self.selected_road_featID = None
        
        # 解析ステップ用
        self.analysis_steps = []
        self.current_step = 0
        
        # シミュレーション状態
        self.path = []  # 各点は {'x':..., 'y':...}（度単位）
        self.running = False
        self.animation_id = None
        self.dt = 0.1
        
        # トラックのパラメータ（単位: m）
        self.vehicle_params = {
            'wheelbase': 4.0,
            'width': 2.5,
            'front_overhang': 1.0,
            'rear_overhang': 1.0,
            'max_steering': math.radians(45),
            'speed': 5.0,
            'lookahead': 10.0
        }
        # 内部計算用（度単位）
        self.wheelBase_m = self.vehicle_params['wheelbase']
        self.frontOverhang_m = self.vehicle_params['front_overhang']
        self.rearOverhang_m = self.vehicle_params['rear_overhang']
        self.vehicleWidth_m = self.vehicle_params['width']
        self.maxSteeringAngle_deg = math.degrees(self.vehicle_params['max_steering'])
        self.vehicleSpeed_m_s = self.vehicle_params['speed']
        self.lookahead_m = self.vehicle_params['lookahead']

        self.wheelBase_deg = 0
        self.frontOverhang_deg = 0
        self.rearOverhang_deg = 0
        self.vehicleWidth_deg = 0
        self.maxSteeringAngle = 0
        self.vehicleSpeed_deg_s = 0
        self.lookahead_deg = 0

        # トラック状態（モデル座標、度単位）
        self.truck_x = 0
        self.truck_y = 0
        self.truck_theta = 0
        self.truck_velocity = 0
        self.truck_steering = 0
        
        self.corner_trajs = [[], [], [], []]
        self.history = []
        
        # 表示変換パラメータ
        self.margin = 20
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.rotation = 0.0
        
        # マウス操作
        self.bind("<Button-1>", self.on_left_click)
        self.bind("<ButtonPress-2>", self.on_mid_down)
        self.bind("<B2-Motion>", self.on_mid_drag)
        self.bind("<ButtonRelease-2>", self.on_mid_up)
        self.bind("<MouseWheel>", self.on_mousewheel)
        
        # 分析用Matplotlib表示
        self.fig = Figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.mpl_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 解析ステップの定義
        self.add_analysis_step("道路読み込み", self.load_and_parse_geojson)
        self.add_analysis_step("経路生成", self.generate_path)
        self.add_analysis_step("衝突検出", self.check_collisions)
        self.add_analysis_step("安全性評価", self.evaluate_safety)
        
        # Performance data（プレースホルダー）
        self.performance_data = {
            'cpu': [],
            'memory': [],
            'timestamps': []
        }
        
    # ----- 解析ステップ管理 -----
    def add_analysis_step(self, name, func):
        self.analysis_steps.append((name, func))
        
    def run_analysis(self):
        for step_name, step_func in self.analysis_steps:
            self.show_progress(f"実行中: {step_name}")
            result = step_func()
            self.visualize_step(result)
            time.sleep(1)
        self.show_progress("解析完了")
        
    def show_progress(self, message):
        if hasattr(self.master, 'status_label'):
            self.master.status_label.config(text=message)
        self.update_idletasks()
        
    def visualize_step(self, data):
        self.ax.clear()
        # 道路描画
        if data.get('road'):
            road = data['road']
            x, y = zip(*road)
            self.ax.plot(x, y, 'b-', linewidth=2)
            self.ax.set_title("道路形状")
        # 経路描画
        if data.get('path'):
            path = np.array(data['path'])
            self.ax.plot(path[:,0], path[:,1], 'g--', linewidth=1)
        # 衝突点描画
        if data.get('collisions'):
            for pt in data['collisions']:
                self.ax.plot(pt[0], pt[1], 'rx', markersize=10)
        # 安全性評価表示
        if data.get('safety') is not None:
            self.ax.text(0.05, 0.95, f"安全性: {data['safety']:.2f}",
                         transform=self.ax.transAxes, va='top')
        self.mpl_canvas.draw()
    
    # ----- シミュレーション用メソッド（従来の GeoJsonTruckCanvas 機能） -----
    def load_and_parse_geojson(self):
        # ファイル選択ダイアログでGeoJSONを読み込み、道路のLineStringとroadWidthMを取得
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson")])
        if not fp:
            return {}
        with open(fp, "r", encoding="utf-8") as f:
            self.geojson_data = json.load(f)
        self.extract_geo_points()
        self.compute_bounds()
        self.full_view()
        self.redraw()
        
        road_line = []
        road_width = 3.0
        for feature in self.geojson_data['features']:
            if feature['geometry']['type'] == 'LineString':
                road_line = feature['geometry']['coordinates']
                props = feature.get('properties', {})
                road_width = props.get('roadWidthM', 3.0)
                break  # 最初のLineStringのみ使用
        # バッファ生成（半幅）
        buffer_distance = road_width / 2.0
        self.road_buffer = LineString(road_line).buffer(buffer_distance)
        return {'road': road_line, 'width': road_width}
        
    def generate_path(self):
        # 単純な経路生成（ここでは、最初のLineStringの全座標をそのままパスとする）
        path = []
        for feature in self.geojson_data['features']:
            if feature['geometry']['type'] == 'LineString':
                for coord in feature['geometry']['coordinates']:
                    path.append({'x': coord[0], 'y': coord[1]})
                break
        self.path = path
        return {'path': self.path}
    
    def check_collisions(self):
        collision_points = []
        for pose in self.path:
            vehicle_poly = self.calculate_vehicle_footprint(pose)
            if not self.road_buffer.contains(vehicle_poly):
                centroid = vehicle_poly.centroid
                collision_points.append((centroid.x, centroid.y))
        return {'collisions': collision_points}
    
    def evaluate_safety(self):
        safety_margin = self.calculate_safety_margin()
        return {'safety': safety_margin}
    
    def calculate_vehicle_footprint(self, pose):
        x, y, theta = pose.get('x',0), pose.get('y',0), pose.get('theta', 0)
        L = self.vehicle_params['wheelbase']
        W = self.vehicle_params['width']
        a = self.vehicle_params['front_overhang']
        b = self.vehicle_params['rear_overhang']
        corners = [(a, W/2), (a, -W/2), (-b, -W/2), (-b, W/2)]
        rotated = [
            (x + dx*math.cos(theta) - dy*math.sin(theta),
             y + dx*math.sin(theta) + dy*math.cos(theta))
            for dx,dy in corners
        ]
        return Polygon(rotated)
    
    def calculate_safety_margin(self):
        # 仮の安全性計算（実際の計算ロジックは用途に合わせて実装）
        return 0.8

    # ----- シミュレーション（従来機能） -----
    def deg_per_m(self):
        if not self.bounds:
            return 1e-6
        (min_lon, max_lon, min_lat, max_lat) = self.bounds
        lat_c = (min_lat+max_lat)*0.5
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(lat_c))
        meter_per_deg = (lat_m+lon_m)*0.5
        return 1.0 / meter_per_deg

    def deg_to_px(self, deg_val):
        x1, y1 = self.model_to_canvas(0,0)
        x2, y2 = self.model_to_canvas(deg_val,0)
        return max(1, math.hypot(x2-x1, y2-y1))

    def model_to_canvas(self, lon, lat):
        cosr = math.cos(self.rotation)
        sinr = math.sin(self.rotation)
        rx = lon*cosr - lat*sinr
        ry = lon*sinr + lat*cosr
        sx = rx*self.scale_factor
        sy = ry*self.scale_factor
        cx = sx + self.offset_x
        cy = -sy + self.offset_y
        return (cx, cy)

    def canvas_to_model(self, cx, cy):
        x1 = cx - self.offset_x
        y1 = -(cy - self.offset_y)
        if self.scale_factor==0:
            self.scale_factor = 1
        rx = x1/self.scale_factor
        ry = y1/self.scale_factor
        r = -self.rotation
        cosr = math.cos(r)
        sinr = math.sin(r)
        mx = rx*cosr - ry*sinr
        my = rx*sinr + ry*cosr
        return (mx,my)

    def on_left_click(self, event):
        mx, my = event.x, event.y
        (lon, lat) = self.canvas_to_model(mx, my)
        snap_px = 10
        best = (lon,lat)
        best_d = 999999
        for (gx,gy) in self.geo_points:
            vx, vy = self.model_to_canvas(gx,gy)
            dd = math.hypot(vx-mx, vy-my)
            if dd < snap_px and dd < best_d:
                best_d = dd
                best = (gx,gy)
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
        self.offset_x = (self.offset_x-cx)*factor+cx
        self.offset_y = (self.offset_y-cy)*factor+cy
        self.scale_factor *= factor
        self.redraw()

    def update_truck_scale(self):
        if not self.bounds:
            return
        (min_lon, max_lon, min_lat, max_lat) = self.bounds
        lat_c = (min_lat+max_lat)*0.5
        lat_m = 111320.0
        lon_m = 111320.0*math.cos(math.radians(lat_c))
        meter_per_deg = (lat_m+lon_m)*0.5
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
            # スタート位置を「トラックのケツ」（後部）とし、そこから全長分（前OH+HB+後OH）だけ前方にずらす
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
        if not self.running:
            self.check_vehicle_fit(during_sim=False)
            return

        tgt = self.find_lookahead_target()
        if not tgt:
            self.running = False
            self.check_vehicle_fit(during_sim=False)
            return

        dx = tgt['x'] - self.truck_x
        dy = tgt['y'] - self.truck_y
        alpha = math.atan2(dy, dx) - self.truck_theta
        alpha = normalize_angle(alpha)

        L = self.wheelBase_deg
        scmd = math.atan2(2 * L * math.sin(alpha), self.lookahead_deg)
        scmd = max(-self.maxSteeringAngle, min(self.maxSteeringAngle, scmd))
        self.truck_steering = scmd

        ang_vel = (self.truck_velocity / L) * math.tan(self.truck_steering)
        self.truck_theta += ang_vel * self.dt
        self.truck_theta = normalize_angle(self.truck_theta)
        self.truck_x += self.truck_velocity * self.dt * math.cos(self.truck_theta)
        self.truck_y += self.truck_velocity * self.dt * math.sin(self.truck_theta)

        if abs(ang_vel) > 0.1:
            self.truck_velocity = max(self.vehicleSpeed_deg_s * 0.5, self.truck_velocity - 0.0005)
        else:
            self.truck_velocity = min(self.vehicleSpeed_deg_s, self.truck_velocity + 0.0005)

        self.update_vehicle_state()
        self.redraw()

        dist_ = math.hypot(self.path[-1]['x'] - self.truck_x, self.path[-1]['y'] - self.truck_y)
        if dist_ < self.lookahead_deg * 0.5:
            self.running = False

        if self.running:
            self.animation_id = self.after(int(self.dt*1000), self.sim_step)
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
                    dd = math.hypot(ip['x']-cx, ip['y']-cy)
                    if dd < best_d:
                        best_d = dd
                        best = ip
        if not best and len(self.path) > 0:
            best = self.path[-1]
        return best

    def circle_line_intersect(self, cx, cy, r, p1, p2):
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        a = dx*dx + dy*dy
        b = 2*(dx*(p1['x']-cx) + dy*(p1['y']-cy))
        c = (p1['x']-cx)**2 + (p1['y']-cy)**2 - r*r
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        sd = math.sqrt(disc)
        t1 = (-b + sd) / (2*a)
        t2 = (-b - sd) / (2*a)
        out = []
        for t in (t1, t2):
            if 0 <= t <= 1:
                ix = p1['x'] + t*dx
                iy = p1['y'] + t*dy
                out.append({'x': ix, 'y': iy})
        return out if out else None

    def check_vehicle_fit(self, during_sim=False):
        """
        シミュレーション終了時に、スタートからの移動距離が車両全長の50%以上の場合、
        トラックの四隅が対象道路のバッファ（Shapely）内に含まれているかチェックします。
        すべて含まれていれば搬入成功、1点でも外なら搬入不可と判定します。
        """
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
        feat = feats[0]  # 対象道路は先頭フィーチャ
        geom = feat["geometry"]
        if geom["type"] != "LineString":
            return False
        props = feat.setdefault("properties", {})
        roadWidthM = props.get("roadWidthM", 2.0)
        half_width_m = roadWidthM / 2.0
        
        # Shapelyによるバッファポリゴンを生成
        deg_per_m = self.deg_per_m()
        buffer_deg = half_width_m * deg_per_m
        road_line = LineString(geom["coordinates"])
        road_buffer = road_line.buffer(buffer_deg)
        
        # トラックの四隅（Point）
        truck_corners = []
        halfW = self.vehicleWidth_deg * 0.5
        totalLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {'x': 0, 'y': +halfW},
            {'x': 0, 'y': -halfW},
            {'x': -totalLen, 'y': -halfW},
            {'x': -totalLen, 'y': +halfW}
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

    def update_vehicle_state(self):
        halfW = self.vehicleWidth_deg * 0.5
        totalLen = self.frontOverhang_deg + self.wheelBase_deg + self.rearOverhang_deg
        corners_local = [
            {'x': 0, 'y': +halfW},       # Front Right
            {'x': 0, 'y': -halfW},       # Front Left
            {'x': -totalLen, 'y': -halfW}, # Rear Left
            {'x': -totalLen, 'y': +halfW}  # Rear Right
        ]
        th = self.truck_theta
        for i, c in enumerate(corners_local):
            rx = c['x'] * math.cos(th) - c['y'] * math.sin(th)
            ry = c['x'] * math.sin(th) + c['y'] * math.cos(th)
            wx = self.truck_x + rx
            wy = self.truck_y + ry
            if i < len(self.corner_trajs):
                self.corner_trajs[i].append({'x': wx, 'y': wy})
        self.history.append({'x': self.truck_x, 'y': self.truck_y, 'theta': self.truck_theta})

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
            {'x': -totalLen, 'y': +halfW}
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
                    dd = math.hypot(ip['x']-cx, ip['y']-cy)
                    if dd < best_d:
                        best_d = dd
                        best = ip
        if not best and len(self.path) > 0:
            best = self.path[-1]
        return best

    def circle_line_intersect(self, cx, cy, r, p1, p2):
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        a = dx*dx + dy*dy
        b = 2 * (dx*(p1['x']-cx) + dy*(p1['y']-cy))
        c = (p1['x']-cx)**2 + (p1['y']-cy)**2 - r*r
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        sd = math.sqrt(disc)
        t1 = (-b+sd)/(2*a)
        t2 = (-b-sd)/(2*a)
        out = []
        for t in (t1, t2):
            if 0 <= t <= 1:
                ix = p1['x'] + t*dx
                iy = p1['y'] + t*dy
                out.append({'x': ix, 'y': iy})
        return out if out else None

###############################################
# MainApplication (AdvancedMainApp)
###############################################
class AdvancedMainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Simulation System")
        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)
        
        # GeoJSON, 回転, トラックパラメータ, シミュレーション操作
        load_fr = ttk.Frame(top_fr)
        load_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_fr, text="GeoJSON読み込み", command=self.on_load).pack(fill=tk.X, pady=2)
        ttk.Button(load_fr, text="全体表示", command=self.on_fit).pack(fill=tk.X, pady=2)
        
        rot_fr = ttk.LabelFrame(top_fr, text="回転", padding=5)
        rot_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_fr, text="←(-10°)", command=lambda: self.on_rotate(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_fr, text="→(+10°)", command=lambda: self.on_rotate(10)).pack(side=tk.LEFT, padx=2)
        
        sep = ttk.Separator(top_fr, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        param_fr = ttk.LabelFrame(top_fr, text="トラックパラメータ", padding=5)
        param_fr.pack(side=tk.LEFT, padx=5)
        self.entries = {}
        def addparam(lbl, defv, key):
            rfr = ttk.Frame(param_fr)
            rfr.pack(anchor="w", pady=2)
            ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
            e = ttk.Entry(rfr, width=6)
            e.insert(0, defv)
            e.pack(side=tk.LEFT)
            self.entries[key] = e
        addparam("WB(m):", "4.0", "wb")
        addparam("前OH(m):", "1.0", "fo")
        addparam("後OH(m):", "1.0", "ro")
        addparam("幅(m):", "2.5", "w")
        addparam("ステア角(°):", "45", "steer")
        addparam("速度(m/s):", "5", "spd")
        addparam("ルック(m):", "10", "look")
        
        sim_fr = ttk.LabelFrame(top_fr, text="シミュレーション", padding=5)
        sim_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_fr, text="リセット", command=self.on_reset).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="スタート", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="一時停止", command=self.on_pause).pack(fill=tk.X, pady=2)
        
        main_fr = ttk.Frame(root)
        main_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = IntegratedGeoJsonTruckCanvas(main_fr)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
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
        
        # Advanced Visualization ボタン
        adv_toolbar = ttk.Frame(root, padding=5)
        adv_toolbar.pack(fill=tk.X)
        ttk.Button(adv_toolbar, text="3D Analysis", command=self.show_advanced_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Scenario Manager", command=self.open_scenario_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Load AI Model", command=self.load_ai_model).pack(side=tk.LEFT, padx=5)
        
        # Performance Monitor パネル
        self.perf_panel = ttk.LabelFrame(root, text="Performance Monitor")
        self.perf_panel.pack(fill=tk.X)
        self.cpu_label = ttk.Label(self.perf_panel, text="CPU: --%")
        self.cpu_label.pack(side=tk.LEFT, padx=5)
        self.gpu_label = ttk.Label(self.perf_panel, text="GPU: --%")
        self.gpu_label.pack(side=tk.LEFT, padx=5)
        self.mem_label = ttk.Label(self.perf_panel, text="Memory: --%")
        self.mem_label.pack(side=tk.LEFT, padx=5)
        
        # パフォーマンス更新は簡易的にタイマーで更新（ここではデモ用）
        self.update_perf()

    def update_perf(self):
        # ここではpsutilを使って簡易的に更新
        self.cpu_label.config(text=f"CPU: {psutil.cpu_percent()}%")
        mem = psutil.virtual_memory().percent
        self.mem_label.config(text=f"Memory: {mem}%")
        self.after(1000, self.update_perf)

    def on_load(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All Files", "*.*")])
        if fp:
            # 解析用のロードも実施
            self.canvas.load_and_parse_geojson()
            self.populate_road_list()

    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()

    def on_rotate(self, deg_):
        r = math.radians(deg_)
        self.canvas.rotation = normalize_angle(self.canvas.rotation + r)
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

    def show_advanced_visualization(self):
        if not hasattr(self, 'adv_vis'):
            vis_window = tk.Toplevel(self.root)
            self.adv_vis = AdvancedVisualization(vis_window)
        visualization_data = {
            'timestamps': self.canvas.performance_data.get('timestamps', []),
            'cpu': self.canvas.performance_data.get('cpu', []),
            'memory': self.canvas.performance_data.get('memory', []),
            'trajectory': {
                'x': [p['x'] for p in self.canvas.history],
                'y': [p['y'] for p in self.canvas.history]
            },
            'speed_profile': [0]*len(self.canvas.history)
        }
        self.adv_vis.update_plots(visualization_data)

    def open_scenario_manager(self):
        ScenarioManagementDialog(self.root, self.canvas)

    def load_ai_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            model_id = self.canvas.load_custom_model(model_path)
            if model_id:
                messagebox.showinfo("Success", f"Model loaded with ID: {model_id}")

###############################################
# Advanced Visualization
###############################################
class AdvancedVisualization:
    def __init__(self, parent):
        self.parent = parent
        self.fig = Figure(figsize=(12,8))
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232, projection='3d')
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def update_plots(self, data):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        self.ax1.plot(data['timestamps'], data['cpu'], label='CPU')
        self.ax1.plot(data['timestamps'], data['memory'], label='Memory')
        self.ax1.set_title("System Performance")
        
        self.ax2.plot(data['trajectory']['x'], data['trajectory']['y'], 
                      np.arange(len(data['trajectory']['x'])),
                      c='blue')
        self.ax2.set_title("3D Trajectory")
        
        hist, xedges, yedges = np.histogram2d(data['trajectory']['x'], data['trajectory']['y'], bins=20)
        self.ax3.imshow(hist.T, origin='lower')
        self.ax3.set_title("Position Heatmap")
        
        self.ax4.plot(data['speed_profile'], label='Speed')
        self.ax4.set_title("Speed Profile")
        
        self.canvas.draw()

###############################################
# Scenario Management Dialog
###############################################
class ScenarioManagementDialog(tk.Toplevel):
    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("Scenario Management")
        self.geometry("800x600")
        self.tree = ttk.Treeview(self, columns=('id','name','created'), show='headings')
        self.tree.heading('id', text='ID')
        self.tree.heading('name', text='Name')
        self.tree.heading('created', text='Created At')
        self.tree.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Load", command=self.load_scenario).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Compare", command=self.compare_scenarios).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Export", command=self.export_scenario).pack(side=tk.LEFT)
        self.refresh_list()
    def refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for scenario in self.system.scenarios.values():
            self.tree.insert('', 'end', values=(
                scenario['id'],
                scenario['name'],
                scenario['created_at'].strftime("%Y-%m-%d %H:%M")
            ))
    def load_scenario(self):
        selected = self.tree.selection()
        if not selected:
            return
        scenario_id = self.tree.item(selected[0])['values'][0]
        scenario = self.system.scenarios[scenario_id]
        messagebox.showinfo("Loaded", f"Scenario {scenario['name']} loaded")
    def compare_scenarios(self):
        pass
    def export_scenario(self):
        pass

###############################################
# MainApplication (AdvancedMainApp)
###############################################
class AdvancedMainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Simulation System")
        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)
        
        load_fr = ttk.Frame(top_fr)
        load_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_fr, text="GeoJSON読み込み", command=self.on_load).pack(fill=tk.X, pady=2)
        ttk.Button(load_fr, text="全体表示", command=self.on_fit).pack(fill=tk.X, pady=2)
        
        rot_fr = ttk.LabelFrame(top_fr, text="回転", padding=5)
        rot_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_fr, text="←(-10°)", command=lambda: self.on_rotate(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_fr, text="→(+10°)", command=lambda: self.on_rotate(10)).pack(side=tk.LEFT, padx=2)
        
        sep = ttk.Separator(top_fr, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        param_fr = ttk.LabelFrame(top_fr, text="トラックパラメータ", padding=5)
        param_fr.pack(side=tk.LEFT, padx=5)
        self.entries = {}
        def addparam(lbl, defv, key):
            rfr = ttk.Frame(param_fr)
            rfr.pack(anchor="w", pady=2)
            ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
            e = ttk.Entry(rfr, width=6)
            e.insert(0, defv)
            e.pack(side=tk.LEFT)
            self.entries[key] = e
        addparam("WB(m):", "4.0", "wb")
        addparam("前OH(m):", "1.0", "fo")
        addparam("後OH(m):", "1.0", "ro")
        addparam("幅(m):", "2.5", "w")
        addparam("ステア角(°):", "45", "steer")
        addparam("速度(m/s):", "5", "spd")
        addparam("ルック(m):", "10", "look")
        
        sim_fr = ttk.LabelFrame(top_fr, text="シミュレーション", padding=5)
        sim_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_fr, text="リセット", command=self.on_reset).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="スタート", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="一時停止", command=self.on_pause).pack(fill=tk.X, pady=2)
        
        main_fr = ttk.Frame(root)
        main_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = IntegratedGeoJsonTruckCanvas(main_fr)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
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
        
        adv_toolbar = ttk.Frame(root, padding=5)
        adv_toolbar.pack(fill=tk.X)
        ttk.Button(adv_toolbar, text="3D Analysis", command=self.show_advanced_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Scenario Manager", command=self.open_scenario_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Load AI Model", command=self.load_ai_model).pack(side=tk.LEFT, padx=5)
        
        self.perf_panel = ttk.LabelFrame(root, text="Performance Monitor")
        self.perf_panel.pack(fill=tk.X)
        self.cpu_label = ttk.Label(self.perf_panel, text="CPU: --%")
        self.cpu_label.pack(side=tk.LEFT, padx=5)
        self.mem_label = ttk.Label(self.perf_panel, text="Memory: --%")
        self.mem_label.pack(side=tk.LEFT, padx=5)
        
        # 簡易パフォーマンス更新
        self.update_perf()

    def update_perf(self):
        self.cpu_label.config(text=f"CPU: {psutil.cpu_percent()}%")
        mem = psutil.virtual_memory().percent
        self.mem_label.config(text=f"Memory: {mem}%")
        self.root.after(1000, self.update_perf)

    def on_load(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All Files", "*.*")])
        if fp:
            self.canvas.load_and_parse_geojson()
            self.populate_road_list()

    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()

    def on_rotate(self, deg_):
        r = math.radians(deg_)
        self.canvas.rotation = normalize_angle(self.canvas.rotation + r)
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

    def show_advanced_visualization(self):
        if not hasattr(self, 'adv_vis'):
            vis_window = tk.Toplevel(self.root)
            self.adv_vis = AdvancedVisualization(vis_window)
        visualization_data = {
            'timestamps': self.canvas.performance_data.get('timestamps', []),
            'cpu': self.canvas.performance_data.get('cpu', []),
            'memory': self.canvas.performance_data.get('memory', []),
            'trajectory': {
                'x': [p['x'] for p in self.canvas.history],
                'y': [p['y'] for p in self.canvas.history]
            },
            'speed_profile': [0]*len(self.canvas.history)
        }
        self.adv_vis.update_plots(visualization_data)

    def open_scenario_manager(self):
        ScenarioManagementDialog(self.root, self.canvas)

    def load_ai_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            model_id = self.canvas.load_custom_model(model_path)
            if model_id:
                messagebox.showinfo("Success", f"Model loaded with ID: {model_id}")

###############################################
# Advanced Visualization
###############################################
class AdvancedVisualization:
    def __init__(self, parent):
        self.parent = parent
        self.fig = Figure(figsize=(12,8))
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232, projection='3d')
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def update_plots(self, data):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        self.ax1.plot(data['timestamps'], data['cpu'], label='CPU')
        self.ax1.plot(data['timestamps'], data['memory'], label='Memory')
        self.ax1.set_title("System Performance")
        
        self.ax2.plot(data['trajectory']['x'], data['trajectory']['y'], 
                      np.arange(len(data['trajectory']['x'])), c='blue')
        self.ax2.set_title("3D Trajectory")
        
        hist, xedges, yedges = np.histogram2d(data['trajectory']['x'], data['trajectory']['y'], bins=20)
        self.ax3.imshow(hist.T, origin='lower')
        self.ax3.set_title("Position Heatmap")
        
        self.ax4.plot(data['speed_profile'], label='Speed')
        self.ax4.set_title("Speed Profile")
        self.canvas.draw()

###############################################
# Scenario Management Dialog
###############################################
class ScenarioManagementDialog(tk.Toplevel):
    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("Scenario Management")
        self.geometry("800x600")
        self.tree = ttk.Treeview(self, columns=('id','name','created'), show='headings')
        self.tree.heading('id', text='ID')
        self.tree.heading('name', text='Name')
        self.tree.heading('created', text='Created At')
        self.tree.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Load", command=self.load_scenario).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Compare", command=self.compare_scenarios).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Export", command=self.export_scenario).pack(side=tk.LEFT)
        self.refresh_list()
    def refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for scenario in self.system.scenarios.values():
            self.tree.insert('', 'end', values=(
                scenario['id'],
                scenario['name'],
                scenario['created_at'].strftime("%Y-%m-%d %H:%M")
            ))
    def load_scenario(self):
        selected = self.tree.selection()
        if not selected:
            return
        scenario_id = self.tree.item(selected[0])['values'][0]
        scenario = self.system.scenarios[scenario_id]
        messagebox.showinfo("Loaded", f"Scenario {scenario['name']} loaded")
    def compare_scenarios(self):
        pass
    def export_scenario(self):
        pass

###############################################
# IntegratedGeoJsonTruckCanvas: シミュレーション＋解析統合クラス
###############################################
class IntegratedGeoJsonTruckCanvas(IntegratedGeoJsonTruckCanvas) if False else IntegratedGeoJsonTruckCanvas(IntegratedGeoJsonTruckCanvas):
    # ※ここでは、上記 EnhancedGeoJsonTruckCanvas の解析機能と従来のシミュレーション機能を統合しています。
    def __init__(self, parent):
        super().__init__(parent)
        # 統合した解析ステップはすでに add_analysis_step() で追加済みです.
        
    def load_custom_model(self, model_path):
        try:
            model = torch.jit.load(model_path)
            model_id = str(uuid4())
            # モデルを内部に保存（以降の処理はプレースホルダー）
            self.custom_models = getattr(self, 'custom_models', {})
            self.custom_models[model_id] = model
            return model_id
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return None

###############################################
# AdvancedMainApp: 統合システム用メインアプリケーション
###############################################
class AdvancedMainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Simulation System")
        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)
        load_fr = ttk.Frame(top_fr)
        load_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_fr, text="GeoJSON読み込み", command=self.on_load).pack(fill=tk.X, pady=2)
        ttk.Button(load_fr, text="全体表示", command=self.on_fit).pack(fill=tk.X, pady=2)
        rot_fr = ttk.LabelFrame(top_fr, text="回転", padding=5)
        rot_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_fr, text="←(-10°)", command=lambda: self.on_rotate(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_fr, text="→(+10°)", command=lambda: self.on_rotate(10)).pack(side=tk.LEFT, padx=2)
        sep = ttk.Separator(top_fr, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        param_fr = ttk.LabelFrame(top_fr, text="トラックパラメータ", padding=5)
        param_fr.pack(side=tk.LEFT, padx=5)
        self.entries = {}
        def addparam(lbl, defv, key):
            rfr = ttk.Frame(param_fr)
            rfr.pack(anchor="w", pady=2)
            ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
            e = ttk.Entry(rfr, width=6)
            e.insert(0, defv)
            e.pack(side=tk.LEFT)
            self.entries[key] = e
        addparam("WB(m):", "4.0", "wb")
        addparam("前OH(m):", "1.0", "fo")
        addparam("後OH(m):", "1.0", "ro")
        addparam("幅(m):", "2.5", "w")
        addparam("ステア角(°):", "45", "steer")
        addparam("速度(m/s):", "5", "spd")
        addparam("ルック(m):", "10", "look")
        sim_fr = ttk.LabelFrame(top_fr, text="シミュレーション", padding=5)
        sim_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_fr, text="リセット", command=self.on_reset).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="スタート", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="一時停止", command=self.on_pause).pack(fill=tk.X, pady=2)
        main_fr = ttk.Frame(root)
        main_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = IntegratedGeoJsonTruckCanvas(main_fr)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
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
        adv_toolbar = ttk.Frame(root, padding=5)
        adv_toolbar.pack(fill=tk.X)
        ttk.Button(adv_toolbar, text="3D Analysis", command=self.show_advanced_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Scenario Manager", command=self.open_scenario_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Load AI Model", command=self.load_ai_model).pack(side=tk.LEFT, padx=5)
        self.perf_panel = ttk.LabelFrame(root, text="Performance Monitor")
        self.perf_panel.pack(fill=tk.X)
        self.cpu_label = ttk.Label(self.perf_panel, text="CPU: --%")
        self.cpu_label.pack(side=tk.LEFT, padx=5)
        self.mem_label = ttk.Label(self.perf_panel, text="Memory: --%")
        self.mem_label.pack(side=tk.LEFT, padx=5)
        self.update_perf()

    def update_perf(self):
        self.cpu_label.config(text=f"CPU: {psutil.cpu_percent()}%")
        mem = psutil.virtual_memory().percent
        self.mem_label.config(text=f"Memory: {mem}%")
        self.root.after(1000, self.update_perf)

    def on_load(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All Files", "*.*")])
        if fp:
            self.canvas.load_and_parse_geojson()
            self.populate_road_list()

    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()

    def on_rotate(self, deg_):
        r = math.radians(deg_)
        self.canvas.rotation = normalize_angle(self.canvas.rotation + r)
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

    def show_advanced_visualization(self):
        if not hasattr(self, 'adv_vis'):
            vis_window = tk.Toplevel(self.root)
            self.adv_vis = AdvancedVisualization(vis_window)
        visualization_data = {
            'timestamps': self.canvas.performance_data.get('timestamps', []),
            'cpu': self.canvas.performance_data.get('cpu', []),
            'memory': self.canvas.performance_data.get('memory', []),
            'trajectory': {
                'x': [p['x'] for p in self.canvas.history],
                'y': [p['y'] for p in self.canvas.history]
            },
            'speed_profile': [0]*len(self.canvas.history)
        }
        self.adv_vis.update_plots(visualization_data)

    def open_scenario_manager(self):
        ScenarioManagementDialog(self.root, self.canvas)

    def load_ai_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            model_id = self.canvas.load_custom_model(model_path)
            if model_id:
                messagebox.showinfo("Success", f"Model loaded with ID: {model_id}")

###############################################
# Advanced Visualization
###############################################
class AdvancedVisualization:
    def __init__(self, parent):
        self.parent = parent
        self.fig = Figure(figsize=(12,8))
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232, projection='3d')
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def update_plots(self, data):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        self.ax1.plot(data['timestamps'], data['cpu'], label='CPU')
        self.ax1.plot(data['timestamps'], data['memory'], label='Memory')
        self.ax1.set_title("System Performance")
        
        self.ax2.plot(data['trajectory']['x'], data['trajectory']['y'],
                      np.arange(len(data['trajectory']['x'])), c='blue')
        self.ax2.set_title("3D Trajectory")
        
        hist, xedges, yedges = np.histogram2d(data['trajectory']['x'], data['trajectory']['y'], bins=20)
        self.ax3.imshow(hist.T, origin='lower')
        self.ax3.set_title("Position Heatmap")
        
        self.ax4.plot(data['speed_profile'], label='Speed')
        self.ax4.set_title("Speed Profile")
        self.canvas.draw()

###############################################
# Scenario Management Dialog
###############################################
class ScenarioManagementDialog(tk.Toplevel):
    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("Scenario Management")
        self.geometry("800x600")
        self.tree = ttk.Treeview(self, columns=('id','name','created'), show='headings')
        self.tree.heading('id', text='ID')
        self.tree.heading('name', text='Name')
        self.tree.heading('created', text='Created At')
        self.tree.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Load", command=self.load_scenario).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Compare", command=self.compare_scenarios).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Export", command=self.export_scenario).pack(side=tk.LEFT)
        self.refresh_list()
    def refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for scenario in self.system.scenarios.values():
            self.tree.insert('', 'end', values=(
                scenario['id'],
                scenario['name'],
                scenario['created_at'].strftime("%Y-%m-%d %H:%M")
            ))
    def load_scenario(self):
        selected = self.tree.selection()
        if not selected:
            return
        scenario_id = self.tree.item(selected[0])['values'][0]
        scenario = self.system.scenarios[scenario_id]
        messagebox.showinfo("Loaded", f"Scenario {scenario['name']} loaded")
    def compare_scenarios(self):
        pass
    def export_scenario(self):
        pass

###############################################
# IntegratedGeoJsonTruckCanvas を AdvancedSimulationSystem として扱う
###############################################
class AdvancedSimulationSystem(IntegratedGeoJsonTruckCanvas):
    # すでに統合済みの機能をそのまま利用
    pass

###############################################
# MainApplication の拡張（AdvancedMainApp）
###############################################
class AdvancedMainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Simulation System")
        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)
        load_fr = ttk.Frame(top_fr)
        load_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_fr, text="GeoJSON読み込み", command=self.on_load).pack(fill=tk.X, pady=2)
        ttk.Button(load_fr, text="全体表示", command=self.on_fit).pack(fill=tk.X, pady=2)
        rot_fr = ttk.LabelFrame(top_fr, text="回転", padding=5)
        rot_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_fr, text="←(-10°)", command=lambda: self.on_rotate(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_fr, text="→(+10°)", command=lambda: self.on_rotate(10)).pack(side=tk.LEFT, padx=2)
        sep = ttk.Separator(top_fr, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        param_fr = ttk.LabelFrame(top_fr, text="トラックパラメータ", padding=5)
        param_fr.pack(side=tk.LEFT, padx=5)
        self.entries = {}
        def addparam(lbl, defv, key):
            rfr = ttk.Frame(param_fr)
            rfr.pack(anchor="w", pady=2)
            ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
            e = ttk.Entry(rfr, width=6)
            e.insert(0, defv)
            e.pack(side=tk.LEFT)
            self.entries[key] = e
        addparam("WB(m):", "4.0", "wb")
        addparam("前OH(m):", "1.0", "fo")
        addparam("後OH(m):", "1.0", "ro")
        addparam("幅(m):", "2.5", "w")
        addparam("ステア角(°):", "45", "steer")
        addparam("速度(m/s):", "5", "spd")
        addparam("ルック(m):", "10", "look")
        sim_fr = ttk.LabelFrame(top_fr, text="シミュレーション", padding=5)
        sim_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_fr, text="リセット", command=self.on_reset).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="スタート", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="一時停止", command=self.on_pause).pack(fill=tk.X, pady=2)
        main_fr = ttk.Frame(root)
        main_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = AdvancedSimulationSystem(main_fr)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
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
        adv_toolbar = ttk.Frame(root, padding=5)
        adv_toolbar.pack(fill=tk.X)
        ttk.Button(adv_toolbar, text="3D Analysis", command=self.show_advanced_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Scenario Manager", command=self.open_scenario_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Load AI Model", command=self.load_ai_model).pack(side=tk.LEFT, padx=5)
        self.perf_panel = ttk.LabelFrame(root, text="Performance Monitor")
        self.perf_panel.pack(fill=tk.X)
        self.cpu_label = ttk.Label(self.perf_panel, text="CPU: --%")
        self.cpu_label.pack(side=tk.LEFT, padx=5)
        self.mem_label = ttk.Label(self.perf_panel, text="Memory: --%")
        self.mem_label.pack(side=tk.LEFT, padx=5)
        self.update_perf()
        
    def update_perf(self):
        self.cpu_label.config(text=f"CPU: {psutil.cpu_percent()}%")
        mem = psutil.virtual_memory().percent
        self.mem_label.config(text=f"Memory: {mem}%")
        self.root.after(1000, self.update_perf)
        
    def on_load(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All Files", "*.*")])
        if fp:
            self.canvas.load_and_parse_geojson()
            self.populate_road_list()
        
    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()
        
    def on_rotate(self, deg_):
        r = math.radians(deg_)
        self.canvas.rotation = normalize_angle(self.canvas.rotation + r)
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
        
    def show_advanced_visualization(self):
        if not hasattr(self, 'adv_vis'):
            vis_window = tk.Toplevel(self.root)
            self.adv_vis = AdvancedVisualization(vis_window)
        visualization_data = {
            'timestamps': self.canvas.performance_data.get('timestamps', []),
            'cpu': self.canvas.performance_data.get('cpu', []),
            'memory': self.canvas.performance_data.get('memory', []),
            'trajectory': {
                'x': [p['x'] for p in self.canvas.history],
                'y': [p['y'] for p in self.canvas.history]
            },
            'speed_profile': [0]*len(self.canvas.history)
        }
        self.adv_vis.update_plots(visualization_data)
        
    def open_scenario_manager(self):
        ScenarioManagementDialog(self.root, self.canvas)
        
    def load_ai_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            model_id = self.canvas.load_custom_model(model_path)
            if model_id:
                messagebox.showinfo("Success", f"Model loaded with ID: {model_id}")
                
###############################################
# Advanced Visualization
###############################################
class AdvancedVisualization:
    def __init__(self, parent):
        self.parent = parent
        self.fig = Figure(figsize=(12,8))
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232, projection='3d')
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def update_plots(self, data):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        self.ax1.plot(data['timestamps'], data['cpu'], label='CPU')
        self.ax1.plot(data['timestamps'], data['memory'], label='Memory')
        self.ax1.set_title("System Performance")
        
        self.ax2.plot(data['trajectory']['x'], data['trajectory']['y'], 
                      np.arange(len(data['trajectory']['x'])), c='blue')
        self.ax2.set_title("3D Trajectory")
        
        hist, xedges, yedges = np.histogram2d(data['trajectory']['x'], data['trajectory']['y'], bins=20)
        self.ax3.imshow(hist.T, origin='lower')
        self.ax3.set_title("Position Heatmap")
        
        self.ax4.plot(data['speed_profile'], label='Speed')
        self.ax4.set_title("Speed Profile")
        self.canvas.draw()
        
###############################################
# Scenario Management Dialog
###############################################
class ScenarioManagementDialog(tk.Toplevel):
    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("Scenario Management")
        self.geometry("800x600")
        self.tree = ttk.Treeview(self, columns=('id','name','created'), show='headings')
        self.tree.heading('id', text='ID')
        self.tree.heading('name', text='Name')
        self.tree.heading('created', text='Created At')
        self.tree.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Load", command=self.load_scenario).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Compare", command=self.compare_scenarios).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Export", command=self.export_scenario).pack(side=tk.LEFT)
        self.refresh_list()
    def refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for scenario in self.system.scenarios.values():
            self.tree.insert('', 'end', values=(
                scenario['id'],
                scenario['name'],
                scenario['created_at'].strftime("%Y-%m-%d %H:%M")
            ))
    def load_scenario(self):
        selected = self.tree.selection()
        if not selected:
            return
        scenario_id = self.tree.item(selected[0])['values'][0]
        scenario = self.system.scenarios[scenario_id]
        messagebox.showinfo("Loaded", f"Scenario {scenario['name']} loaded")
    def compare_scenarios(self):
        pass
    def export_scenario(self):
        pass

###############################################
# AdvancedSimulationSystem
###############################################
class AdvancedSimulationSystem(IntegratedGeoJsonTruckCanvas):
    # AdvancedSimulationSystem は IntegratedGeoJsonTruckCanvas をそのまま利用します。
    # ここで追加の機能が必要ならオーバーライドしてください。
    pass

###############################################
# Main Application
###############################################
class AdvancedMainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Simulation System")
        top_fr = ttk.Frame(root, padding=5)
        top_fr.pack(side=tk.TOP, fill=tk.X)
        
        load_fr = ttk.Frame(top_fr)
        load_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(load_fr, text="GeoJSON読み込み", command=self.on_load).pack(fill=tk.X, pady=2)
        ttk.Button(load_fr, text="全体表示", command=self.on_fit).pack(fill=tk.X, pady=2)
        
        rot_fr = ttk.LabelFrame(top_fr, text="回転", padding=5)
        rot_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_fr, text="←(-10°)", command=lambda: self.on_rotate(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_fr, text="→(+10°)", command=lambda: self.on_rotate(10)).pack(side=tk.LEFT, padx=2)
        
        sep = ttk.Separator(top_fr, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        param_fr = ttk.LabelFrame(top_fr, text="トラックパラメータ", padding=5)
        param_fr.pack(side=tk.LEFT, padx=5)
        self.entries = {}
        def addparam(lbl, defv, key):
            rfr = ttk.Frame(param_fr)
            rfr.pack(anchor="w", pady=2)
            ttk.Label(rfr, text=lbl).pack(side=tk.LEFT)
            e = ttk.Entry(rfr, width=6)
            e.insert(0, defv)
            e.pack(side=tk.LEFT)
            self.entries[key] = e
        addparam("WB(m):", "4.0", "wb")
        addparam("前OH(m):", "1.0", "fo")
        addparam("後OH(m):", "1.0", "ro")
        addparam("幅(m):", "2.5", "w")
        addparam("ステア角(°):", "45", "steer")
        addparam("速度(m/s):", "5", "spd")
        addparam("ルック(m):", "10", "look")
        
        sim_fr = ttk.LabelFrame(top_fr, text="シミュレーション", padding=5)
        sim_fr.pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_fr, text="リセット", command=self.on_reset).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="スタート", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(sim_fr, text="一時停止", command=self.on_pause).pack(fill=tk.X, pady=2)
        
        main_fr = ttk.Frame(root)
        main_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = AdvancedSimulationSystem(main_fr)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        roadlist_fr = ttk.LabelFrame(main_fr, text="道路リスト(選択でハイライト)", padding=5)
        roadlist_fr.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.road_tree = ttk.Treeview(roadlist_fr, columns=("featID","roadWidthM"), show="headings", height=15)
        self.road_tree.heading("featID", text="ID")
        self.road_tree.heading("roadWidthM", text="幅(m)")
        self.road_tree.column("featID", width=50)
        self.road_tree.column("roadWidthM", width=80)
        self.road_tree.pack(side=tk.TOP, fill=tk.Y, expand=True)
        apply_btn = ttk.Button(roadlist_fr, text="適用", command=self.on_road_apply)
        apply_btn.pack(side=tk.TOP, pady=8)
        self.road_tree.bind("<<TreeviewSelect>>", self.on_road_select)
        self.road_tree.bind("<Double-1>", self.on_road_dblclick)
        
        adv_toolbar = ttk.Frame(root, padding=5)
        adv_toolbar.pack(fill=tk.X)
        ttk.Button(adv_toolbar, text="3D Analysis", command=self.show_advanced_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Scenario Manager", command=self.open_scenario_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(adv_toolbar, text="Load AI Model", command=self.load_ai_model).pack(side=tk.LEFT, padx=5)
        
        self.perf_panel = ttk.LabelFrame(root, text="Performance Monitor")
        self.perf_panel.pack(fill=tk.X)
        self.cpu_label = ttk.Label(self.perf_panel, text="CPU: --%")
        self.cpu_label.pack(side=tk.LEFT, padx=5)
        self.mem_label = ttk.Label(self.perf_panel, text="Memory: --%")
        self.mem_label.pack(side=tk.LEFT, padx=5)
        self.update_perf()
        
        # Advanced Analysis 起動ボタン
        analysis_btn = ttk.Button(root, text="解析開始", command=self.canvas.run_analysis)
        analysis_btn.pack(side=tk.BOTTOM, pady=5)
        
        # シナリオ管理領域（プレースホルダー）
        self.canvas.scenarios = {}
        
    def update_perf(self):
        self.cpu_label.config(text=f"CPU: {psutil.cpu_percent()}%")
        mem = psutil.virtual_memory().percent
        self.mem_label.config(text=f"Memory: {mem}%")
        self.root.after(1000, self.update_perf)
        
    def on_load(self):
        fp = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson"), ("All Files", "*.*")])
        if fp:
            self.canvas.load_and_parse_geojson()
            self.populate_road_list()
        
    def on_fit(self):
        self.canvas.full_view()
        self.canvas.redraw()
        
    def on_rotate(self, deg_):
        r = math.radians(deg_)
        self.canvas.rotation = normalize_angle(self.canvas.rotation + r)
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
        
    def show_advanced_visualization(self):
        if not hasattr(self, 'adv_vis'):
            vis_window = tk.Toplevel(self.root)
            self.adv_vis = AdvancedVisualization(vis_window)
        visualization_data = {
            'timestamps': self.canvas.performance_data.get('timestamps', []),
            'cpu': self.canvas.performance_data.get('cpu', []),
            'memory': self.canvas.performance_data.get('memory', []),
            'trajectory': {
                'x': [p['x'] for p in self.canvas.history],
                'y': [p['y'] for p in self.canvas.history]
            },
            'speed_profile': [0]*len(self.canvas.history)
        }
        self.adv_vis.update_plots(visualization_data)
        
    def open_scenario_manager(self):
        ScenarioManagementDialog(self.root, self.canvas)
        
    def load_ai_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            model_id = self.canvas.load_custom_model(model_path)
            if model_id:
                messagebox.showinfo("Success", f"Model loaded with ID: {model_id}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedMainApp(root)
    root.mainloop()

###############################################
# 以下は追加機能のプレースホルダー
###############################################
def auto_generate_initial_path(self):
    """道路データから自動的に初期経路を生成"""
    path = []
    for feature in self.geojson_data['features']:
        if feature['geometry']['type'] == 'LineString':
            for coord in feature['geometry']['coordinates']:
                path.append({'x': coord[0], 'y': coord[1]})
            break
    return self.smooth_path(path)

def autonomous_optimization_loop(self):
    while True:
        self.path = self.auto_generate_initial_path()
        self.advanced_ai_optimization()
        result = self.run_simulation()
        if result['success_rate'] > 0.95:
            self.generate_report(result)
            break
        else:
            self.adjust_parameters()

def multi_objective_optimization(self):
    objectives = {
        'safety': self.calculate_safety_score(),
        'efficiency': self.calculate_time_efficiency(),
        'comfort': self.calculate_ride_comfort()
    }
    return self.nsga2_optimizer.optimize(objectives)

def realtime_adaptation(self):
    while self.running:
        current_state = self.get_sensor_data()
        optimal_action = self.predict_optimal_action(current_state)
        self.apply_control(optimal_action)
        time.sleep(0.1)

class ExplainableAI:
    def explain_decision(self, decision):
        return {
            'factors': self.analyze_decision_factors(decision),
            'confidence_level': self.calculate_confidence(),
            'alternative_options': self.generate_alternatives()
        }
