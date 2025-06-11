# main_window.py (AttributeError修正最終版)

import tkinter as tk
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from tkinter import ttk, filedialog, messagebox, Listbox, Scrollbar, Frame
import math
import webbrowser
import json
import config
import geojson_io
import osm_data
from vehicle import TruckModel, VehicleState
from simulation import SimulationController, normalize_angle
from canvas_view import SimulationView
import sys
import traceback
from shapely.geometry import Point, LineString
import requests

class MainWindow(tk.Frame):
    """メインアプリケーションウィンドウ (道路選択->端点クリック経路作成)"""

    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.pack(fill=tk.BOTH, expand=True)
        self.root.title("トラック走行シミュレータ (経路:端点クリック)")

        # --- データ関連 ---
        self.current_osm_data_m: dict | None = None
        self.current_bounds_m: tuple | None = None
        self.simulation_path_m: list[dict] = []

        # --- 状態管理 ---
        self.current_mode: str = "view"
        self.selected_osm_ids: list[int] = []
        self.selected_road_features: list[dict] = []
        self.selectable_endpoints: list[tuple] = []

        # --- モデル・シミュレーション ---
        self.truck_config: dict = {}
        self.truck_model: TruckModel | None = None
        self.simulation_controller: SimulationController | None = None

        # --- UI要素 ---
        self.canvas_view: SimulationView | None = None
        self.param_entries: dict[str, tk.StringVar] = {}
        self.status_vars: dict[str, tk.StringVar] = {}
        self.btn_start: ttk.Button | None = None
        self.btn_pause: ttk.Button | None = None
        self.btn_reset: ttk.Button | None = None
        self.btn_start_path_definition: ttk.Button | None = None
        self.btn_clear_path: ttk.Button | None = None
        self.mode_var = tk.StringVar(value="view")
        self.latlon_input_var = tk.StringVar()
        self.selected_roads_listbox: Listbox | None = None
        self.path_listbox: Listbox | None = None
        self.mode_status_var = tk.StringVar(value="モード: ビュー操作")
        self.redraw_timer_id = None
        self.btn_move_up: ttk.Button | None = None
        self.btn_move_down: ttk.Button | None = None
        self.btn_remove_road: ttk.Button | None = None

        # --- UI構築 ---
        self.build_ui(self) # メソッド名を build_ui に変更
        self._load_initial_config()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def build_ui(self, parent_frame): # _setup_ui からリネーム
        """UI要素の配置"""
        top_frame = ttk.Frame(parent_frame); top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        sidebar_frame = ttk.Frame(parent_frame, width=350); sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5); sidebar_frame.pack_propagate(False)
        canvas_frame = ttk.Frame(parent_frame); canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        try:
            self.canvas_view = SimulationView(canvas_frame); self.canvas_view.pack(fill=tk.BOTH, expand=True)
            if hasattr(self.canvas_view, "set_master_controller"): self.canvas_view.set_master_controller(self)
            else: print("Warn: SimulationView has no set_master_controller method.")
        except Exception as e: messagebox.showerror("致命的エラー", f"Canvas初期化失敗:\n{e}"); print(traceback.format_exc()); self.root.destroy(); return
        self._setup_top_buttons(top_frame)
        self._setup_sidebar(sidebar_frame)
        self._update_ui_states()

    def _setup_top_buttons(self, parent_frame):
        """トップフレームのボタン類"""
        data_group = ttk.LabelFrame(parent_frame, text="OSMデータ"); data_group.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.Y)
        f1 = ttk.Frame(data_group); f1.pack(pady=2, anchor=tk.W)
        ttk.Label(f1, text="中心(緯,経):").pack(side=tk.LEFT); self.latlon_input_var.set("35.6812, 139.7671"); e1 = ttk.Entry(f1, textvariable=self.latlon_input_var, width=20); e1.pack(side=tk.LEFT, padx=2); e1.bind("<Return>", lambda e: self.on_fetch_osm_data_from_input())
        ttk.Button(data_group, text="周辺取得", command=self.on_fetch_osm_data_from_input).pack(pady=2, fill=tk.X, padx=3)

        view_group = ttk.LabelFrame(parent_frame, text="ビュー"); view_group.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.Y)
        ttk.Button(view_group, text="全体", command=self.on_fit_view).pack(side=tk.LEFT, padx=2, pady=5)
        ttk.Button(view_group, text="←", width=3, command=lambda: self.on_rotate(-10)).pack(side=tk.LEFT, padx=2, pady=5)
        ttk.Button(view_group, text="→", width=3, command=lambda: self.on_rotate(10)).pack(side=tk.LEFT, padx=2, pady=5)

        mode_group = ttk.LabelFrame(parent_frame, text="モード/経路"); mode_group.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.Y)
        f2 = ttk.Frame(mode_group); f2.pack(anchor=tk.W)
        ttk.Radiobutton(f2, text="ビュー", variable=self.mode_var, value="view", command=self._on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(f2, text="道路選択", variable=self.mode_var, value="select_road", command=self._on_mode_change).pack(side=tk.LEFT)
        ttk.Label(mode_group, textvariable=self.mode_status_var, width=22).pack(anchor=tk.W, padx=5, pady=2)
        f3 = ttk.Frame(mode_group); f3.pack(anchor=tk.W, fill=tk.X)
        self.btn_start_path_definition = ttk.Button(f3, text="経路定義開始", command=self.on_start_path_definition, state=tk.DISABLED); self.btn_start_path_definition.pack(side=tk.LEFT, padx=5, pady=(2,5))
        self.btn_clear_path = ttk.Button(f3, text="経路クリア", command=self.on_clear_path, state=tk.DISABLED); self.btn_clear_path.pack(side=tk.LEFT, padx=5, pady=(2,5))

    def _setup_sidebar(self, parent_frame):
        """サイドバーの要素"""
        param_label=ttk.Label(parent_frame,text="トラックパラメータ",font=("",10,"bold")); param_label.pack(pady=(5,5),anchor=tk.W,padx=5)
        param_frame=ttk.Frame(parent_frame); param_frame.pack(fill=tk.X,padx=5)
        def add_param(lbl,key,val): f=ttk.Frame(param_frame);f.pack(fill=tk.X,pady=1);ttk.Label(f,text=lbl,width=16,anchor=tk.W).pack(side=tk.LEFT);v=tk.StringVar(value=str(val));e=ttk.Entry(f,textvariable=v,width=10);e.pack(side=tk.LEFT,expand=True,fill=tk.X);e.bind("<FocusOut>",lambda ev:self._apply_truck_params());e.bind("<Return>",lambda ev:self._apply_truck_params());self.param_entries[key]=v
        add_param("ホイールベース(m):",'wheelBase_m',config.DEFAULT_WHEEL_BASE_M); add_param("前OH(m):",'frontOverhang_m',config.DEFAULT_FRONT_OVERHANG_M)
        add_param("後OH(m):",'rearOverhang_m',config.DEFAULT_REAR_OVERHANG_M); add_param("車体幅(m):",'vehicleWidth_m',config.DEFAULT_VEHICLE_WIDTH_M)
        add_param("最大舵角(度):",'maxSteering_deg',config.DEFAULT_MAX_STEERING_DEG); add_param("目標速度(m/s):",'targetSpeed_mps',config.DEFAULT_VEHICLE_SPEED_MPS)
        add_param("ルックアヘッド(m):",'lookahead_m',config.DEFAULT_LOOKAHEAD_M)
        ttk.Separator(parent_frame,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5,padx=5)

        sel_lbl=ttk.Label(parent_frame,text="選択道路リスト",font=("",10,"bold")); sel_lbl.pack(pady=(0,5),anchor=tk.W,padx=5)
        sel_frame=ttk.Frame(parent_frame); sel_frame.pack(fill=tk.X,padx=5)
        sel_btn_frame=ttk.Frame(sel_frame); sel_btn_frame.pack(side=tk.TOP,fill=tk.X,pady=(0,5))
        self.btn_move_up=ttk.Button(sel_btn_frame,text="↑",width=3,command=self.on_move_up,state=tk.DISABLED); self.btn_move_up.pack(side=tk.LEFT,padx=2)
        self.btn_move_down=ttk.Button(sel_btn_frame,text="↓",width=3,command=self.on_move_down,state=tk.DISABLED); self.btn_move_down.pack(side=tk.LEFT,padx=2)
        self.btn_remove_road=ttk.Button(sel_btn_frame,text="削除",width=5,command=self.on_remove_selected,state=tk.DISABLED); self.btn_remove_road.pack(side=tk.RIGHT,padx=2)
        sel_list_frame=ttk.Frame(sel_frame); sel_list_frame.pack(fill=tk.X)
        sel_scroll=Scrollbar(sel_list_frame,orient=tk.VERTICAL)
        self.selected_roads_listbox=Listbox(sel_list_frame,yscrollcommand=sel_scroll.set,height=5,selectmode=tk.SINGLE,exportselection=False); sel_scroll.config(command=self.selected_roads_listbox.yview); sel_scroll.pack(side=tk.RIGHT,fill=tk.Y); self.selected_roads_listbox.pack(side=tk.LEFT,fill=tk.X,expand=True)
        self.selected_roads_listbox.bind("<<ListboxSelect>>", lambda e: self._update_ui_states())

        ttk.Separator(parent_frame,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5,padx=5)
        path_lbl=ttk.Label(parent_frame,text="作成経路 (端点リスト)",font=("",10,"bold")); path_lbl.pack(pady=(0,5),anchor=tk.W,padx=5)
        path_list_frame=ttk.Frame(parent_frame); path_list_frame.pack(fill=tk.BOTH,expand=True,padx=5,pady=(0,5))
        path_scroll=Scrollbar(path_list_frame,orient=tk.VERTICAL)
        self.path_listbox=Listbox(path_list_frame,yscrollcommand=path_scroll.set,height=6); path_scroll.config(command=self.path_listbox.yview); path_scroll.pack(side=tk.RIGHT,fill=tk.Y); self.path_listbox.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)

        ttk.Separator(parent_frame,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5,padx=5)
        sim_lbl = ttk.Label(parent_frame, text="シミュレーション", font=("", 10, "bold")); sim_lbl.pack(pady=(0, 5), anchor=tk.W, padx=5)
        sim_ctrl_frame=ttk.Frame(parent_frame); sim_ctrl_frame.pack(fill=tk.X,padx=5)
        self.btn_start=ttk.Button(sim_ctrl_frame,text="開始/再開",command=self.on_start_simulation,state=tk.DISABLED); self.btn_start.pack(side=tk.LEFT,padx=2,expand=True,fill=tk.X)
        self.btn_pause=ttk.Button(sim_ctrl_frame,text="一時停止",command=self.on_pause_simulation,state=tk.DISABLED); self.btn_pause.pack(side=tk.LEFT,padx=2,expand=True,fill=tk.X)
        self.btn_reset=ttk.Button(sim_ctrl_frame,text="リセット",command=self.on_reset_simulation,state=tk.DISABLED); self.btn_reset.pack(side=tk.LEFT,padx=2,expand=True,fill=tk.X)

        ttk.Separator(parent_frame,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5,padx=5)
        status_lbl = ttk.Label(parent_frame, text="状態", font=("", 10, "bold")); status_lbl.pack(pady=(0, 5), anchor=tk.W, padx=5)
        status_frame = ttk.Frame(parent_frame); status_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        def add_status(lbl,key): f=ttk.Frame(status_frame);f.pack(fill=tk.X,pady=1);ttk.Label(f,text=lbl,width=12,anchor=tk.W).pack(side=tk.LEFT);v=tk.StringVar(value="-");ttk.Label(f,textvariable=v,anchor=tk.W).pack(side=tk.LEFT,expand=True,fill=tk.X);self.status_vars[key]=v
        add_status("X座標(m):",'x_m');add_status("Y座標(m):",'y_m');add_status("向き(度):",'theta_deg');add_status("速度(m/s):",'velocity_mps');add_status("舵角(度):",'steering_deg')

    def _load_initial_config(self):
        print("Loading config...")
        self._update_mode_display()
        try:
            geojson_io._initialize_transformers()
            if not self._apply_truck_params():
                messagebox.showwarning("初期化", "パラメータ適用失敗")
        except Exception as e:
            messagebox.showerror("初期化エラー", f"エラー:\n{e}")
            print(traceback.format_exc())
            self.root.destroy()
            return
        print("Config loaded.")

    def _apply_truck_params(self) -> bool:
        current_config={};
        try:
            for k,v in self.param_entries.items(): current_config[k]=float(v.get())
            self.truck_config=current_config; print(f"Applying params: {self.truck_config}")
        except Exception as e: messagebox.showerror("入力エラー",f"パラメータ無効:\n{e}"); return False
        try:
            self.truck_model=TruckModel(self.truck_config)
            if self.canvas_view and hasattr(self.canvas_view,"set_truck_model"):
                 self.canvas_view.set_truck_model(self.truck_model)
                 if self.canvas_view.winfo_exists(): self.canvas_view.redraw()
            if self.simulation_controller:
                self.simulation_controller.lookahead_m=self.truck_config.get('lookahead_m',config.DEFAULT_LOOKAHEAD_M)
                self.simulation_controller.target_velocity_mps=self.truck_config.get('targetSpeed_mps',config.DEFAULT_VEHICLE_SPEED_MPS)
                self.simulation_controller.truck=self.truck_model; print("SimController updated.")
            self._update_status_display(); return True
        except Exception as e: messagebox.showerror("エラー",f"パラメータ適用エラー:\n{e}"); print(traceback.format_exc()); return False

    def _update_status_display(self):
        st={}
        if self.truck_model:
            try: s=self.truck_model.state; st['x_m']=f"{s.x_m:.2f}"; st['y_m']=f"{s.y_m:.2f}"; st['theta_deg']=f"{math.degrees(s.theta_rad):.1f}"; st['velocity_mps']=f"{s.velocity_mps:.2f}"; st['steering_deg']=f"{math.degrees(s.steering_rad):.1f}"
            except Exception as e: print(f"Err get state:{e}")
        else:
            for k in self.status_vars: st[k]="-"
        for k,v in self.status_vars.items():
            try:
                if self.root.winfo_exists(): v.set(st.get(k,"-"))
            except tk.TclError: pass

    def _on_mode_change(self):
        new_mode = self.mode_var.get(); print(f"Mode radio -> {new_mode}")
        if self.current_mode == 'define_path_by_endpoints' and new_mode != 'define_path_by_endpoints': print("Exiting path def mode."); self._reset_path_definition_state()
        elif self.current_mode == 'select_road' and new_mode == 'view':
             if self.canvas_view and hasattr(self.canvas_view,'selected_osm_way_ids'): self.canvas_view.selected_osm_way_ids=set(); self.canvas_view.redraw()
        elif new_mode == 'define_path_by_endpoints': print("Warn: Use button."); self.mode_var.set(self.current_mode); return
        self.current_mode = new_mode
        if self.canvas_view and hasattr(self.canvas_view,'set_mode'): self.canvas_view.set_mode(self.current_mode)
        self._update_mode_display(); self._update_ui_states()

    def _update_mode_display(self):
        mode_text="モード: ";
        if self.current_mode=='select_road': mode_text+="道路選択"
        elif self.current_mode=='define_path_by_endpoints': mode_text+="経路定義 (端点クリック)"
        else: mode_text+="ビュー操作"
        try:
             if self.root.winfo_exists(): self.mode_status_var.set(mode_text)
        except tk.TclError: pass

    def _reset_path_definition_state(self):
        print("Resetting path def state."); self.selectable_endpoints=[]
        if self.canvas_view and hasattr(self.canvas_view,'clear_all_endpoint_markers'):
            try:
                 if self.canvas_view.winfo_exists(): self.canvas_view.clear_all_endpoint_markers(); self.canvas_view.redraw()
            except tk.TclError: pass
        if self.current_mode=='define_path_by_endpoints':
             self.current_mode='select_road'
             if self.mode_var.get()!='view': self.mode_var.set('select_road')
             if self.canvas_view and hasattr(self.canvas_view,'set_mode'): self.canvas_view.set_mode(self.current_mode)
             self._update_mode_display()
        self._update_ui_states()

    def on_clear_path(self):
        if not self.simulation_path_m and self.current_mode!='define_path_by_endpoints': print("Nothing to clear."); return
        if messagebox.askyesno("確認","作成中経路削除しますか？"):
            print("Clearing path."); self.simulation_path_m = []
            if self.path_listbox:
                 try:
                      if self.path_listbox.winfo_exists(): self.path_listbox.delete(0,tk.END)
                 except tk.TclError: pass
            if self.current_mode=='define_path_by_endpoints': self._reset_path_definition_state()
            if self.simulation_controller:
                if self.simulation_controller.running: self._stop_redraw_loop(); self.simulation_controller.pause()
                self.simulation_controller=None
            self._update_ui_states()
            if self.canvas_view: self.canvas_view.redraw()

    def on_fetch_osm_data_from_input(self):
        input_str=self.latlon_input_var.get();
        if not input_str: messagebox.showwarning("入力","座標未入力"); return
        try: parts=input_str.split(','); lat=float(parts[0].strip()); lon=float(parts[1].strip()); assert -90<=lat<=90 and -180<=lon<=180
        except Exception as e: messagebox.showerror("入力エラー",f"座標形式無効:\n{e}"); return
        delta=config.OSM_FETCH_DEGREE_DELTA; bounds=(lat-delta/2,lon-delta/2,lat+delta/2,lon+delta/2)
        print(f"Request OSM for {bounds}"); self.root.config(cursor="watch"); self.root.update_idletasks()
        try:
            osm_data_m, bounds_m = osm_data.get_osm_data_as_meter_geojson(bounds_latlon=bounds)
            if osm_data_m is None: return
            elif not osm_data_m.get("features"): messagebox.showinfo("情報","データなし"); self.current_osm_data_m=osm_data_m; self.current_bounds_m=None; self._clear_all_data(); return
            self.current_osm_data_m=osm_data_m; self.current_bounds_m=bounds_m
            if self.canvas_view: self.canvas_view.set_osm_data(self.current_osm_data_m, self.current_bounds_m); self.canvas_view.fit_to_screen()
            self._clear_all_data(); print("OSM loaded.")
        except Exception as e: messagebox.showerror("エラー",f"OSM取得エラー:\n{e}"); print(traceback.format_exc())
        finally:
             try:
                  if self.root.winfo_exists(): self.root.config(cursor="")
             except: pass

    def _clear_all_data(self):
         print("Clearing all non-OSM data.")
         self.selected_osm_ids=[]; self.selected_road_features=[]; self.simulation_path_m=[]; self.selectable_endpoints=[]
         if self.selected_roads_listbox:
              try:
                   if self.selected_roads_listbox.winfo_exists(): self.selected_roads_listbox.delete(0,tk.END)
              except tk.TclError: pass
         if self.path_listbox:
              try:
                   if self.path_listbox.winfo_exists(): self.path_listbox.delete(0,tk.END)
              except tk.TclError: pass
         if self.canvas_view and hasattr(self.canvas_view,"clear_all_endpoint_markers"):
             try:
                 if self.canvas_view.winfo_exists(): self.canvas_view.clear_all_endpoint_markers(); self.canvas_view.selected_osm_way_ids=set()
             except tk.TclError: pass
         if self.simulation_controller:
             if self.simulation_controller.running: self._stop_redraw_loop(); self.simulation_controller.pause()
             self.simulation_controller=None
         if self.truck_model: self.truck_model.reset()
         if self.current_mode=='define_path_by_endpoints': self._reset_path_definition_state()
         self.current_mode='view';
         try:
             if self.root.winfo_exists(): self.mode_var.set('view')
         except tk.TclError: pass
         if self.canvas_view and hasattr(self.canvas_view,'set_mode'):
              if self.canvas_view.winfo_exists(): self.canvas_view.set_mode(self.current_mode)
         self._update_mode_display(); self._update_ui_states(); self._update_status_display();
         if self.canvas_view:
              if self.canvas_view.winfo_exists(): self.canvas_view.redraw()

    def on_road_selected(self, way_id: int, feature: dict):
        if self.current_mode!='select_road' or not feature: return
        if self.current_mode=='define_path_by_endpoints': messagebox.showinfo("情報","経路定義中は不可"); return
        is_sel = way_id in self.selected_osm_ids
        if is_sel:
            try:
                idx = self.selected_osm_ids.index(way_id)
                self.selected_osm_ids.pop(idx)
                self.selected_road_features.pop(idx)
                if self.canvas_view:
                    self.canvas_view.highlight_road(way_id, False)
                self._update_selected_roads_listbox()
                print(f"Road deselected:{way_id}")
            except ValueError:
                pass
        else:
            self.selected_osm_ids.append(way_id); self.selected_road_features.append(feature)
            if self.canvas_view: self.canvas_view.highlight_road(way_id,True)
            self._update_selected_roads_listbox();
            if self.selected_roads_listbox: self.selected_roads_listbox.see(tk.END); print(f"Road selected:{way_id}")
        self._update_ui_states()

    def _update_selected_roads_listbox(self):
        if not self.selected_roads_listbox: return
        self.selected_roads_listbox.delete(0, tk.END)
        for i,feature in enumerate(self.selected_road_features):
             way_id=self.selected_osm_ids[i]; props=feature.get("properties",{}); d_name=f"{i+1}: Way {way_id}: "+ " ".join(filter(None,[f'"{props.get("name")}"' if props.get("name") else None,f'[{props.get("ref")}]' if props.get("ref") else None,f'({props.get("highway")})' if props.get("highway") else None,'(Oneway)' if props.get("oneway")=="yes" else None])); self.selected_roads_listbox.insert(tk.END,d_name)

    def on_move_up(self):
        if not self.selected_roads_listbox: return; sel=self.selected_roads_listbox.curselection();
        if not sel: return; idx=sel[0]
        if idx>0: self.selected_osm_ids.insert(idx-1,self.selected_osm_ids.pop(idx)); self.selected_road_features.insert(idx-1,self.selected_road_features.pop(idx)); self._update_selected_roads_listbox(); self.selected_roads_listbox.select_set(idx-1); self.selected_roads_listbox.activate(idx-1); self.selected_roads_listbox.see(idx-1); self._update_ui_states()

    def on_move_down(self):
        if not self.selected_roads_listbox: return; sel=self.selected_roads_listbox.curselection();
        if not sel: return; idx=sel[0]
        if idx < len(self.selected_osm_ids)-1: self.selected_osm_ids.insert(idx+1,self.selected_osm_ids.pop(idx)); self.selected_road_features.insert(idx+1,self.selected_road_features.pop(idx)); self._update_selected_roads_listbox(); self.selected_roads_listbox.select_set(idx+1); self.selected_roads_listbox.activate(idx+1); self.selected_roads_listbox.see(idx+1); self._update_ui_states()

    def on_remove_selected(self):
        if not self.selected_roads_listbox: return; sel=self.selected_roads_listbox.curselection();
        if not sel: return; idx=sel[0]; removed_id=self.selected_osm_ids.pop(idx); self.selected_road_features.pop(idx); self._update_selected_roads_listbox();
        if self.canvas_view: self.canvas_view.highlight_road(removed_id,False)
        if len(self.selected_osm_ids)>0: new_idx=min(idx,len(self.selected_osm_ids)-1); self.selected_roads_listbox.select_set(new_idx); self.selected_roads_listbox.activate(new_idx)
        self._update_ui_states()

    def on_start_path_definition(self):
        if len(self.selected_road_features)<1: messagebox.showwarning("経路定義","道路未選択"); return
        print("Starting path definition..."); endpoints={}
        for feature in self.selected_road_features:
            coords=feature.get("geometry",{}).get("coordinates",[])
            if len(coords)>=2:
                 start_node=tuple(coords[0]); end_node=tuple(coords[-1])
                 if all(isinstance(v,(int,float)) for v in start_node) and all(isinstance(v,(int,float)) for v in end_node): endpoints[start_node]=endpoints.get(start_node,0)+1; endpoints[end_node]=endpoints.get(end_node,0)+1
                 else: print(f"Warn: Invalid coords in WayID {feature.get('properties',{}).get('osm_id')}")
        self.selectable_endpoints = list(endpoints.keys())
        print(f"Found {len(self.selectable_endpoints)} unique endpoints.")
        if not self.selectable_endpoints: messagebox.showerror("エラー","端点抽出失敗"); return
        if self.simulation_path_m:
             if not messagebox.askyesno("確認","既存経路クリアしますか？"): return
             self.on_clear_path()
        self.current_mode='define_path_by_endpoints'; self._update_mode_display()
        if self.canvas_view and hasattr(self.canvas_view,"show_all_selectable_endpoints"):
            self.canvas_view.set_mode(self.current_mode)
            self.canvas_view.show_all_selectable_endpoints(self.selectable_endpoints)
        self._update_ui_states(); messagebox.showinfo("経路定義","端点(赤丸)表示。始点から順にクリック。")

    def on_endpoint_added_to_path(self, coords: tuple):
        if self.current_mode != 'define_path_by_endpoints': return
        print(f"Endpoint ({coords[0]:.2f}, {coords[1]:.2f}) selected."); new_point={'x':coords[0],'y':coords[1]}
        if self.simulation_path_m and abs(self.simulation_path_m[-1]['x']-new_point['x'])<1e-3 and abs(self.simulation_path_m[-1]['y']-new_point['y'])<1e-3: print("Warn: Same point."); return
        self.simulation_path_m.append(new_point)
        if self.path_listbox: idx=len(self.simulation_path_m); self.path_listbox.insert(tk.END,f"{idx}:({coords[0]:.1f},{coords[1]:.1f})"); self.path_listbox.see(tk.END)
        if self.canvas_view and hasattr(self.canvas_view,"mark_endpoint_as_used"): self.canvas_view.mark_endpoint_as_used(coords)
        self._update_ui_states() # 自動開始は削除

    def on_start_simulation(self):
        print("DEBUG: on_start_simulation called.")
        if len(self.simulation_path_m)<2: messagebox.showerror("開始不可","経路点不足"); print("DEBUG: Not enough path points."); return
        if not self.truck_model:
             if not self._apply_truck_params(): print("DEBUG: Truck params apply failed."); return
        if self.current_mode == 'define_path_by_endpoints':
             print("DEBUG: Exiting define_path mode before starting sim.")
             self._reset_path_definition_state(); self.current_mode='view'; self.mode_var.set('view'); self._update_mode_display()
             if self.canvas_view: self.canvas_view.set_mode(self.current_mode)
        try:
            if not self.simulation_controller:
                 print("DEBUG: Init SimController...");
                 try: self.simulation_controller=SimulationController(self.truck_model,self.simulation_path_m); self.simulation_controller.reset(); print("DEBUG: SimController initialized.")
                 except Exception as e: messagebox.showerror("エラー",f"SimController初期化失敗:\n{e}"); print(f"ERROR Init SimController: {e}"); traceback.print_exc(); return
            elif self.simulation_controller.path != self.simulation_path_m: print("DEBUG: Updating path..."); self.simulation_controller.path=self.simulation_path_m; self.simulation_controller.reset(); print("DEBUG: SimController path updated and reset.")
            else: print("DEBUG: Existing SimController will be used.")
            if self.simulation_controller.running: print("DEBUG: Resuming simulation..."); self._start_redraw_loop(); self._update_ui_states(); return
            print("DEBUG: Calling simulation_controller.start()..."); success=self.simulation_controller.start(self.root.after); print(f"DEBUG: simulation_controller.start() returned {success}")
            if success: print("DEBUG: Sim started successfully."); self._start_redraw_loop(); self._update_ui_states()
            else: messagebox.showerror("開始エラー","開始処理失敗 (Controller.start failed)"); print("DEBUG: simulation_controller.start() failed.")
        except Exception as e: messagebox.showerror("実行エラー",f"開始エラー:\n{e}"); print(f"ERROR in on_start_simulation: {e}"); traceback.print_exc(); self._stop_redraw_loop(); self._update_ui_states()
        print("DEBUG: on_start_simulation finished.")

    def on_pause_simulation(self):
        # print("DEBUG: _update_ui_states called.") # on_pauseでは不要かも
        if self.simulation_controller and self.simulation_controller.running: self.simulation_controller.pause(); self._stop_redraw_loop(); self._update_ui_states(); print("Paused.")
        else: print("Not running or no controller.")

    def on_reset_simulation(self):
        if not self.simulation_controller: print("Nothing to reset."); return
        print("Resetting..."); self._stop_redraw_loop(); self.simulation_controller.reset(); self._update_status_display();
        if self.canvas_view: self.canvas_view.redraw();
        self._update_ui_states(); print("Reset done.")

    def _start_redraw_loop(self):
        if self.redraw_timer_id: return; print("Starting redraw..."); self._redraw_loop()
    def _stop_redraw_loop(self):
        if self.redraw_timer_id: print("Stopping redraw..."); 
        try: self.root.after_cancel(self.redraw_timer_id) 
        except: pass; 
        finally: self.redraw_timer_id=None
    def _redraw_loop(self):
        try:
            if not self.root.winfo_exists(): self.redraw_timer_id=None; return; self._update_status_display();
            if self.canvas_view: self.canvas_view.redraw();
            if self.simulation_controller and self.simulation_controller.running: self.redraw_timer_id=self.root.after(config.REDRAW_INTERVAL_MS, self._redraw_loop)
            else: self.redraw_timer_id=None; self._update_ui_states()
        except Exception as e: print(f"Err redraw:{e}"); traceback.print_exc(); self._stop_redraw_loop(); self._update_ui_states(); messagebox.showerror("描画エラー",f"エラー:\n{e}")

    def _update_ui_states(self):
         # print("DEBUG: _update_ui_states called.") # デバッグ用
         if not hasattr(self,'btn_start'): return
         is_view=self.current_mode=='view'; is_sel_road=self.current_mode=='select_road'; is_def_path=self.current_mode=='define_path_by_endpoints'
         has_osm = bool(self.current_osm_data_m and self.current_osm_data_m.get("features"))
         has_road_sel=len(self.selected_osm_ids)>0; has_path=len(self.simulation_path_m)>=2
         ctrl_exists=self.simulation_controller is not None
         # --- ▼▼▼ is_running 判定修正 ▼▼▼ ---
         is_running=False
         if ctrl_exists and hasattr(self.simulation_controller,'running'): is_running=self.simulation_controller.running
         # --- ▲▲▲ 修正 ▲▲▲ ---
         # print(f"DEBUG: UI State - Mode:{self.current_mode}, HasPath:{has_path}, CtrlExists:{ctrl_exists}, IsRunning:{is_running}") # デバッグ用
         try:
              if self.root.winfo_exists():
                   # Top buttons
                   if self.btn_start_path_definition: self.btn_start_path_definition.config(state=tk.NORMAL if has_road_sel and not is_def_path else tk.DISABLED)
                   if self.btn_clear_path: self.btn_clear_path.config(state=tk.NORMAL if self.simulation_path_m or is_def_path else tk.DISABLED)
                   # Sidebar road list buttons
                   list_sel=self.selected_roads_listbox.curselection() if self.selected_roads_listbox else None; idx=list_sel[0] if list_sel else -1
                   state_up=tk.NORMAL if is_sel_road and idx>0 else tk.DISABLED; state_down=tk.NORMAL if is_sel_road and idx>=0 and idx<len(self.selected_osm_ids)-1 else tk.DISABLED; state_remove=tk.NORMAL if is_sel_road and idx>=0 else tk.DISABLED
                   if self.btn_move_up: self.btn_move_up.config(state=state_up)
                   if self.btn_move_down: self.btn_move_down.config(state=state_down)
                   if self.btn_remove_road: self.btn_remove_road.config(state=state_remove)
                   # Simulation buttons
                   # --- ▼▼▼ 開始ボタンの条件修正 ▼▼▼ ---
                   state_start=tk.NORMAL if has_path and not is_running else tk.DISABLED
                   # --- ▲▲▲ 修正 ▲▲▲ ---
                   state_pause=tk.NORMAL if ctrl_exists and is_running else tk.DISABLED
                   state_reset=tk.NORMAL if ctrl_exists or self.simulation_path_m else tk.DISABLED

                   if self.btn_start: self.btn_start.config(state=state_start)
                   if self.btn_pause: self.btn_pause.config(state=state_pause)
                   if self.btn_reset: self.btn_reset.config(state=state_reset)
         except tk.TclError: pass
         # print("DEBUG: _update_ui_states finished.") # デバッグ用

    def on_fit_view(self):
        if self.canvas_view:
            if self.current_osm_data_m and self.current_osm_data_m.get("features"): print("Fitting view."); self.canvas_view.fit_to_screen()
            else: messagebox.showinfo("情報","全体表示データなし")
    def on_rotate(self, angle_deg_delta):
        if self.canvas_view: self.canvas_view.rotate_view(math.radians(angle_deg_delta))

    def _on_closing(self):
        print("Closing..."); self._stop_redraw_loop();
        if self.simulation_controller and self.simulation_controller.running: self.simulation_controller.pause()
        print("Destroying window.")
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        print("Closed.")
# --- MainWindowクラス定義ここまで ---

#canvas_view.pyの修正はいらないよ