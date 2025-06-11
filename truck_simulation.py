# truck_simulation.py

import math
from dataclasses import dataclass
from shapely.geometry import Polygon

def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def normalize_angle(a: float) -> float:
    """角度 a を [-pi, pi] の範囲に正規化する"""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

@dataclass
class TruckState:
    x: float
    y: float
    theta: float     # 車両進行方向（ラジアン）
    velocity: float  # 現在の速度
    steering: float  # 現在のステアリング角（ラジアン）

class TruckSimulation:
    """
    トラックの物理シミュレーションを管理するクラス
    
    ・車両の運動はキネマティック・バイシクルモデルを RK4 法による数値積分で更新
    ・Pure Pursuit 制御により目標点に基づく理想ステアリング角を算出し、
      ステアリング角にレート制限を適用して状態を更新
    ・get_truck_polygon() により、車両の形状（四隅）の Shapely Polygon を返す。
      これを用いて外部から障害物との衝突判定が可能。
    """
    
    def __init__(self,
                 wheel_base_m: float = 4.0,
                 front_overhang_m: float = 1.0,
                 rear_overhang_m: float = 1.0,
                 vehicle_width_m: float = 2.5,
                 max_steering_deg: float = 45.0,
                 vehicle_speed_m_s: float = 5.0,
                 lookahead_m: float = 10.0,
                 dt: float = 0.1):
        self.wheel_base = wheel_base_m
        self.front_overhang = front_overhang_m
        self.rear_overhang = rear_overhang_m
        self.vehicle_width = vehicle_width_m
        self.max_steering = deg_to_rad(max_steering_deg)
        self.vehicle_speed = vehicle_speed_m_s
        self.lookahead = lookahead_m
        self.dt = dt

        # 制御パラメータ
        self.steering_rate_limit = deg_to_rad(20.0)  # 最大ステア角変化速度 [rad/s]
        self.corner_decel = 0.3    # 例：急旋回時の減速率（m/s²）
        self.corner_accel = 0.2    # 例：直進時の加速率（m/s²）

        # 初期状態（パスから初期化する前のダミー値）
        self.state = TruckState(x=0.0, y=0.0, theta=0.0,
                                  velocity=self.vehicle_speed, steering=0.0)

        # ユーザーが設定するパス（リスト：各点は {'x': float, 'y': float}）
        self.path = []

    def set_initial_state_from_path(self):
        """パス上の最初の2点から初期状態（位置と進行方向）を設定する"""
        if len(self.path) >= 2:
            self.state.x = self.path[0]['x']
            self.state.y = self.path[0]['y']
            dx = self.path[1]['x'] - self.path[0]['x']
            dy = self.path[1]['y'] - self.path[0]['y']
            self.state.theta = math.atan2(dy, dx)
        else:
            self.state.x, self.state.y, self.state.theta = 0.0, 0.0, 0.0

    def _compute_desired_steering(self):
        """
        Pure Pursuit 制御で目標点に対応する理想のステアリング角を計算する。
        パス上の目標点は lookahead 距離内のものから選択する。
        """
        L_target = self.lookahead
        best = None
        best_d = float('inf')
        # パス上の線分間の交点を調べる
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i+1]
            inters = self._circle_line_intersect(self.state.x, self.state.y, L_target, p1, p2)
            if inters:
                for pt in inters:
                    # 前方かどうか判定
                    vx = pt['x'] - self.state.x
                    vy = pt['y'] - self.state.y
                    if vx * math.cos(self.state.theta) + vy * math.sin(self.state.theta) < 0:
                        continue
                    dist = math.hypot(vx, vy)
                    if dist < best_d:
                        best_d = dist
                        best = pt
        # 目標が見つからなければパス終端とする
        if best is None and self.path:
            best = self.path[-1]
        if best is None:
            return 0.0
        dx = best['x'] - self.state.x
        dy = best['y'] - self.state.y
        alpha = normalize_angle(math.atan2(dy, dx) - self.state.theta)
        # Pure Pursuit の理論式: δ = arctan(2L*sin(α) / lookahead)
        desired = math.atan2(2 * self.wheel_base * math.sin(alpha), self.lookahead)
        # 制約
        return max(-self.max_steering, min(self.max_steering, desired))

    def _circle_line_intersect(self, cx, cy, r, p1, p2):
        """
        2点 p1, p2 による直線と、中心(cx, cy)、半径 r の円の交点を求める。
        戻り値は交点の辞書リスト [{'x': ..., 'y': ...}, ...]。
        """
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        a = dx**2 + dy**2
        b = 2 * (dx * (p1['x'] - cx) + dy * (p1['y'] - cy))
        c = (p1['x'] - cx)**2 + (p1['y'] - cy)**2 - r**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        intersections = []
        for t in (t1, t2):
            if 0 <= t <= 1:
                intersections.append({'x': p1['x'] + t * dx, 'y': p1['y'] + t * dy})
        return intersections if intersections else None

    def update(self) -> bool:
        """
        トラックの状態を 1 dt だけ更新する。
        ・Pure Pursuit 制御を用いて、理想ステアリング角 desired を求める。
        ・ステアリング角はレート制限を適用して変化させる。
        ・キネマティック・バイシクルモデルの運動方程式を RK4 法で数値積分する。
        ・速度は、旋回時に減速、直線時に加速する処理を適用する（例として）。
        
        戻り値:
          更新に成功すれば True。パスが短い場合など更新不可なら False を返す。
        """
        # パスが2点未満の場合、更新不可
        if len(self.path) < 2:
            return False

        # 目標ステアリング角を計算
        desired_steer = self._compute_desired_steering()

        # ステアリング角のレート制限
        steer_error = desired_steer - self.state.steering
        max_steer_change = self.steering_rate_limit * self.dt
        if abs(steer_error) > max_steer_change:
            steer_error = math.copysign(max_steer_change, steer_error)
        self.state.steering += steer_error

        # RK4 法で状態 (x, y, theta) を更新する
        def dynamics(state, steering):
            # state: [x, y, theta]
            x, y, theta = state
            dx = self.state.velocity * math.cos(theta)
            dy = self.state.velocity * math.sin(theta)
            dtheta = self.state.velocity / self.wheel_base * math.tan(steering)
            return [dx, dy, dtheta]

        s0 = [self.state.x, self.state.y, self.state.theta]
        k1 = dynamics(s0, self.state.steering)
        s1 = [s0[i] + k1[i]*self.dt/2.0 for i in range(3)]
        k2 = dynamics(s1, self.state.steering)
        s2 = [s0[i] + k2[i]*self.dt/2.0 for i in range(3)]
        k3 = dynamics(s2, self.state.steering)
        s3 = [s0[i] + k3[i]*self.dt for i in range(3)]
        k4 = dynamics(s3, self.state.steering)
        new_state = [
            s0[i] + (self.dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            for i in range(3)
        ]
        self.state.x, self.state.y, self.state.theta = new_state
        self.state.theta = normalize_angle(self.state.theta)

        # 速度制御（例: 旋回中は減速、直線中は目標速度に向かって加速）
        if abs(self.state.steering) > 0.1:  # 角度が大きい場合（旋回中）
            self.state.velocity = max(self.vehicle_speed * 0.3, self.state.velocity - self.corner_decel * self.dt)
        else:
            self.state.velocity = min(self.vehicle_speed, self.state.velocity + self.corner_accel * self.dt)

        return True

    def get_truck_polygon(self) -> Polygon:
        """
        現在の状態から、トラックの形状（四隅）の Polygon を Shapely で生成して返す。
        車両前部、車両後部（オーバーハング含む）、横幅に基づいて計算。
        """
        half_w = self.vehicle_width / 2.0
        total_length = self.front_overhang + self.wheel_base + self.rear_overhang
        # ローカル座標（車両前端を0、後ろ方向に total_length）
        local_corners = [
            (0, half_w),
            (0, -half_w),
            (-total_length, -half_w),
            (-total_length, half_w)
        ]
        cos_t = math.cos(self.state.theta)
        sin_t = math.sin(self.state.theta)
        world_corners = []
        for lx, ly in local_corners:
            wx = self.state.x + lx * cos_t - ly * sin_t
            wy = self.state.y + lx * sin_t + ly * cos_t
            world_corners.append((wx, wy))
        return Polygon(world_corners)

    def check_collision(self, obstacles: list[Polygon]) -> bool:
        """
        障害物リスト（Shapely の Polygon のリスト）と現在のトラック形状の交差判定を行う。
        交差していれば True を返す。
        """
        truck_poly = self.get_truck_polygon()
        for obs in obstacles:
            if truck_poly.intersects(obs):
                return True
        return False
