#!/usr/bin/env python3
"""
Simple desktop GUI to select a navigation target, mirroring the way
`keyboard_control` works: run it on your laptop joined to the robot's ROS
master (ROS_MASTER_URI pointing at the robot), and it calls the robot's
`target_selection_node/select_target` service for you — no more attaching to
the container and typing rosservice by hand.

The dropdown is populated from the graph_planner label map (same labels you
would pass to the service). The field is editable, so you can also type a raw
label or node id that is not in the map.
"""
import io
import os
import threading
import tkinter as tk
from tkinter import ttk

import yaml
import rospy
from duckietown_msgs.srv import SetFSMState
from duckietown_msgs.msg import BoolStamped, FSMState
from sensor_msgs.msg import CompressedImage

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class TargetGUINode(object):
    # Duckietown palette
    DUCK_YELLOW      = "#FFCC00"
    DUCK_BLUE        = "#0F6AB4"
    DUCK_BLUE_ACTIVE = "#0C568F"
    DARK             = "#2B2B2B"
    WHITE            = "#FFFFFF"
    # Status colors chosen for contrast on the yellow background
    STATUS_COLORS = {
        "red":   "#B03024",
        "green": "#0B6E2E",
        "blue":  "#0F4C9B",
        "gray":  "#5A5A5A",
        "black": "#2B2B2B",
    }

    def __init__(self):
        rospy.init_node("target_gui_node", anonymous=True)

        self.veh = rospy.get_param(
            "~veh", os.environ.get("VEHICLE_NAME", "myduckiebot"))
        label_map_file = self._resolve_label_map_file()

        self.srv_name = f"/{self.veh}/target_selection_node/select_target"
        self.targets = self._load_targets(label_map_file)

        # Joystick-override toggle. Publishing data=False fires the FSM's
        # joystick_override_off event (NORMAL_JOYSTICK_CONTROL -> IDLE);
        # data=True fires joystick_override_on (global -> NORMAL_JOYSTICK_CONTROL).
        self.autonomous = False   # robot boots in NORMAL_JOYSTICK_CONTROL
        self.pub_override = rospy.Publisher(
            f"/{self.veh}/joy_mapper_node/joystick_override",
            BoolStamped, queue_size=1)

        # Planner intersection-debug toggle + manual GO.
        self.planner_debug = False
        self.pub_debug_mode = rospy.Publisher(
            f"/{self.veh}/graph_planner_node/debug_mode",
            BoolStamped, queue_size=1, latch=True)
        self.pub_debug_go = rospy.Publisher(
            f"/{self.veh}/graph_planner_node/debug_go",
            BoolStamped, queue_size=1)

        # Ground-truth localization toggle — only meaningful in simulation
        # (myduckiebot). The real robot (roboduck) has no ground-truth source.
        self.gt_available = (self.veh == "myduckiebot")
        self.use_ground_truth = False
        self.pub_use_gt = rospy.Publisher(
            f"/{self.veh}/graph_planner_node/use_ground_truth",
            BoolStamped, queue_size=1, latch=True)

        # Live "arrived" feedback from the planner (latched topic).
        rospy.Subscriber(
            f"/{self.veh}/graph_planner_node/arrived_at_target",
            BoolStamped, self._cb_arrived, queue_size=1)

        self._img_photo     = None  # keep PIL references to prevent GC
        self._seg_photo     = None
        self._cluster_photo = None
        self._img_pil_raw   = None  # raw (unscaled) graph planner image
        self._resize_job    = None  # debounce handle for canvas <Configure>

        self._build_ui()

        # Mirror the real FSM state (latched, so we get the current state
        # immediately). Subscribed after the UI exists so the callback can draw.
        rospy.Subscriber(
            f"/{self.veh}/fsm_node/mode",
            FSMState, self._cb_fsm_state, queue_size=1)

        if _PIL_AVAILABLE:
            rospy.Subscriber(
                f"/{self.veh}/graph_planner_node/debug_image/compressed",
                CompressedImage, self._cb_debug_image, queue_size=1)
            rospy.Subscriber(
                f"/{self.veh}/line_detector_node/debug/segments/compressed",
                CompressedImage, self._cb_segments_image, queue_size=1)
            rospy.Subscriber(
                f"/{self.veh}/intersection_type_detector_node/debug/clusters/compressed",
                CompressedImage, self._cb_clusters_image, queue_size=1)
            rospy.Subscriber(
                f"/{self.veh}/camera_node/image/compressed",
                CompressedImage, self._cb_camera_image, queue_size=1,
                buff_size=2 ** 24)
        else:
            rospy.logwarn("[target_gui] Pillow not installed — debug image panels disabled")

    # ---- target list ----

    def _resolve_label_map_file(self):
        """Find the label map without the user having to pass it:
        1) explicit ~label_map_file param, else
        2) whatever the running graph_planner loaded, else
        3) the default file shipped in the navigation package."""
        # 1) explicit override
        path = rospy.get_param("~label_map_file", "")
        if path and os.path.isfile(path):
            return path

        # 2) mirror the running planner (guarantees the GUI matches it)
        try:
            planner_path = rospy.get_param(
                f"/{self.veh}/graph_planner_node/label_map_file")
            if planner_path and os.path.isfile(planner_path):
                rospy.loginfo(f"[target_gui] Using planner's label map: "
                              f"{planner_path}")
                return planner_path
        except KeyError:
            pass  # planner not up yet / param not set

        # 3) default file inside the navigation package (present in this image)
        try:
            import rospkg
            pkg = rospkg.RosPack().get_path("navigation")
            default = os.path.join(pkg, "config", "graph_planner_node",
                                   "label_map_ekf.yaml")
            if os.path.isfile(default):
                rospy.loginfo(f"[target_gui] Using package label map: {default}")
                return default
        except Exception as e:
            rospy.logwarn(f"[target_gui] Could not resolve package path: {e}")

        return path  # possibly "" → empty dropdown, handled by _load_targets

    def _load_targets(self, path):
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                targets = list(data.keys())
                rospy.loginfo(f"[target_gui] Loaded {len(targets)} targets "
                              f"from {path}: {targets}")
                return targets
            except Exception as e:
                rospy.logwarn(f"[target_gui] Could not read label map "
                              f"{path}: {e}")
        else:
            rospy.logwarn(f"[target_gui] label_map_file not found ('{path}') "
                          f"— dropdown will be empty, type targets manually.")
        return []

    # ---- UI ----

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title(f"Target Selection — {self.veh}")
        self.root.minsize(340, 300)
        self.root.configure(bg=self.DUCK_YELLOW)

        pad = dict(padx=14, pady=8)

        r = 0
        self._build_switch(self.root).grid(
            row=r, column=0, columnspan=2, sticky="w", **pad)
        r += 1
        if self.gt_available:
            self._build_gt_switch(self.root).grid(
                row=r, column=0, columnspan=2, sticky="w", **pad)
            r += 1
        self._build_debug_switch(self.root).grid(
            row=r, column=0, columnspan=2, sticky="w", **pad)
        r += 1

        # Manual intersection GO — only visible while planner debug is on.
        self.int_go_btn = tk.Button(self.root, text="▶ Intersection GO",
                                    command=self._send_intersection_go,
                                    bg=self.DUCK_BLUE, fg=self.WHITE,
                                    activebackground=self.DUCK_BLUE_ACTIVE,
                                    activeforeground=self.WHITE,
                                    font=("DejaVu Sans", 12, "bold"),
                                    relief="flat", bd=0, padx=10, pady=6,
                                    cursor="hand2")
        self.int_go_btn.grid(row=r, column=0, columnspan=2, sticky="ew", **pad)
        self.int_go_btn.grid_remove()
        r += 1

        tk.Label(self.root, text="Target location:",
                 bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 12, "bold")).grid(
                     row=r, column=0, columnspan=2, sticky="w", **pad)
        r += 1

        self.target_var = tk.StringVar()
        if self.targets:
            self.target_var.set(self.targets[0])

        # ttk theme "clam" is the one that reliably honors custom colors.
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        # The combobox entry field: blue on white to match the button accent.
        style.configure("Duck.TCombobox",
                        fieldbackground=self.WHITE,
                        background=self.DUCK_BLUE,
                        foreground=self.DUCK_BLUE,
                        arrowcolor=self.WHITE,
                        bordercolor=self.DUCK_BLUE,
                        lightcolor=self.DUCK_BLUE,
                        darkcolor=self.DUCK_BLUE)
        style.map("Duck.TCombobox",
                  fieldbackground=[("readonly", self.WHITE)],
                  foreground=[("readonly", self.DUCK_BLUE)])
        # The drop-down list that pops open (a Tk Listbox under the hood):
        # blue background, white text, matching selection highlight.
        self.root.option_add("*TCombobox*Listbox.background", self.DUCK_BLUE)
        self.root.option_add("*TCombobox*Listbox.foreground", self.WHITE)
        self.root.option_add("*TCombobox*Listbox.selectBackground",
                             self.DUCK_BLUE_ACTIVE)
        self.root.option_add("*TCombobox*Listbox.selectForeground", self.WHITE)
        self.root.option_add("*TCombobox*Listbox.font", "{DejaVu Sans} 11")

        self.combo = ttk.Combobox(self.root, textvariable=self.target_var,
                                  values=self.targets, width=22,
                                  style="Duck.TCombobox",
                                  font=("DejaVu Sans", 11))
        self.combo.grid(row=r, column=0, columnspan=2, sticky="ew", **pad)
        r += 1
        self.combo.bind("<Return>", lambda _e: self._send())

        self.go_btn = tk.Button(self.root, text="GO", command=self._send,
                                bg=self.DUCK_BLUE, fg=self.WHITE,
                                activebackground=self.DUCK_BLUE_ACTIVE,
                                activeforeground=self.WHITE,
                                font=("DejaVu Sans", 13, "bold"),
                                relief="flat", bd=0, padx=10, pady=8,
                                cursor="hand2")
        self.go_btn.grid(row=r, column=0, columnspan=2, sticky="ew", **pad)
        r += 1

        initial = (f"Service: {self.srv_name}" if self.targets
                   else "No label map found — type a label or node ID.")
        self.status_var = tk.StringVar(value=initial)
        self.status_lbl = tk.Label(self.root, textvariable=self.status_var,
                                   bg=self.DUCK_YELLOW, fg=self.DARK,
                                   wraplength=310, justify="left",
                                   font=("DejaVu Sans", 9))
        self.status_lbl.grid(row=r, column=0, columnspan=2, sticky="w", **pad)
        r += 1

        # Debug images: graph planner on the left, segments + clusters stacked on the right.
        # The graph canvas fills and scales with the window; right panel stays fixed width.
        debug_frame = tk.Frame(self.root, bg=self.DUCK_YELLOW)
        debug_frame.grid(row=r, column=0, columnspan=2, padx=14, pady=(4, 4), sticky="nsew")
        self.root.rowconfigure(r, weight=1)
        r += 1

        # ---- Left column: graph planner (expands with window) ----
        left_panel = tk.Frame(debug_frame, bg=self.DUCK_YELLOW)
        left_panel.pack(side="left", anchor="n", padx=(0, 12), fill="both", expand=True)

        tk.Label(left_panel, text="Graph planner:",
                 bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w", pady=(0, 2))

        self._img_canvas = tk.Canvas(left_panel, width=560, height=560,
                                     bg=self.DARK, highlightthickness=0)
        self._img_canvas.pack(fill="both", expand=True)
        self._img_canvas.bind("<Configure>", self._on_img_canvas_resize)
        if not _PIL_AVAILABLE:
            self._img_canvas.create_text(280, 280,
                                         text="Install Pillow to see the\nplanner debug image",
                                         fill="#AAAAAA", font=("DejaVu Sans", 10), justify="center")

        # ---- Right column: camera (top) + segments + clusters (bottom) ----
        right_panel = tk.Frame(debug_frame, bg=self.DUCK_YELLOW)
        right_panel.pack(side="left", anchor="n")

        tk.Label(right_panel, text="Camera:",
                 bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w", pady=(0, 2))

        CAM_W, CAM_H = 560, 420
        self._cam_canvas = tk.Canvas(right_panel, width=CAM_W, height=CAM_H,
                                     bg=self.DARK, highlightthickness=0)
        self._cam_canvas.pack(pady=(0, 10))
        if not _PIL_AVAILABLE:
            self._cam_canvas.create_text(CAM_W // 2, CAM_H // 2,
                                         text="Install Pillow to see the\ncamera image",
                                         fill="#AAAAAA", font=("DejaVu Sans", 10), justify="center")

        tk.Label(right_panel, text="Line detector segments:",
                 bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w", pady=(0, 2))

        SEG_W, SEG_H = 560, 420
        self._seg_canvas = tk.Canvas(right_panel, width=SEG_W, height=SEG_H,
                                     bg=self.DARK, highlightthickness=0)
        self._seg_canvas.pack(pady=(0, 10))
        if not _PIL_AVAILABLE:
            self._seg_canvas.create_text(SEG_W // 2, SEG_H // 2,
                                         text="Install Pillow to see the\nsegment debug image",
                                         fill="#AAAAAA", font=("DejaVu Sans", 10), justify="center")

        tk.Label(right_panel, text="Intersection type detector (red clusters):",
                 bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w", pady=(0, 2))

        CLUSTER_W, CLUSTER_H = 560, 120
        self._cluster_canvas = tk.Canvas(right_panel, width=CLUSTER_W, height=CLUSTER_H,
                                         bg=self.DARK, highlightthickness=0)
        self._cluster_canvas.pack()
        if not _PIL_AVAILABLE:
            self._cluster_canvas.create_text(CLUSTER_W // 2, CLUSTER_H // 2,
                                             text="Install Pillow to see the\ncluster debug image",
                                             fill="#AAAAAA", font=("DejaVu Sans", 10), justify="center")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Shut the GUI down cleanly when ROS dies (e.g. Ctrl-C in the terminal).
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._check_ros)

    # ---- toggle switches ----

    def _draw_pill(self, canvas, on):
        """Draw a rounded pill toggle on `canvas`, knob right (blue) when on."""
        canvas.delete("all")
        w, h = self._sw_w, self._sw_h
        r = h // 2
        track = self.DUCK_BLUE if on else "#B8B8B8"
        canvas.create_oval(2, 2, h - 2, h - 2, fill=track, outline=track)
        canvas.create_oval(w - h + 2, 2, w - 2, h - 2, fill=track, outline=track)
        canvas.create_rectangle(r + 1, 2, w - r - 1, h - 2, fill=track, outline=track)
        knob_r = r - 4
        cx = (w - r) if on else r
        canvas.create_oval(cx - knob_r, r - knob_r, cx + knob_r, r + knob_r,
                           fill=self.WHITE, outline=self.WHITE)

    def _build_switch(self, parent):
        self._sw_w, self._sw_h = 60, 28
        frame = tk.Frame(parent, bg=self.DUCK_YELLOW)
        tk.Label(frame, text="Mode:", bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 12, "bold")).pack(side="left")
        self.mode_canvas = tk.Canvas(frame, width=self._sw_w,
                                     height=self._sw_h, bg=self.DUCK_YELLOW,
                                     highlightthickness=0, cursor="hand2")
        self.mode_canvas.pack(side="left", padx=(8, 8))
        self.mode_canvas.bind("<Button-1>", lambda _e: self._toggle_switch())
        self.mode_text = tk.Label(frame, text="Joystick",
                                  bg=self.DUCK_YELLOW, fg=self.DARK,
                                  font=("DejaVu Sans", 10))
        self.mode_text.pack(side="left")
        self._draw_pill(self.mode_canvas, self.autonomous)
        return frame

    def _build_debug_switch(self, parent):
        frame = tk.Frame(parent, bg=self.DUCK_YELLOW)
        tk.Label(frame, text="Debug:", bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 12, "bold")).pack(side="left")
        self.debug_canvas = tk.Canvas(frame, width=self._sw_w,
                                      height=self._sw_h, bg=self.DUCK_YELLOW,
                                      highlightthickness=0, cursor="hand2")
        self.debug_canvas.pack(side="left", padx=(8, 8))
        self.debug_canvas.bind("<Button-1>", lambda _e: self._toggle_debug())
        self.debug_text = tk.Label(frame, text="off",
                                   bg=self.DUCK_YELLOW, fg=self.DARK,
                                   font=("DejaVu Sans", 10))
        self.debug_text.pack(side="left")
        self._draw_pill(self.debug_canvas, self.planner_debug)
        return frame

    def _toggle_switch(self):
        # A click is a REQUEST; the switch position follows the actual FSM
        # state via _cb_fsm_state, not this click.
        go_autonomous = not self.autonomous
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        # autonomous requested -> override OFF (leave joystick, enter IDLE);
        # joystick requested   -> override ON (global return to joystick).
        msg.data = (not go_autonomous)
        self.pub_override.publish(msg)
        self._set_status("Requested IDLE (autonomous) — select a target."
                         if go_autonomous
                         else "Requested joystick control.", "blue")

    def _build_gt_switch(self, parent):
        frame = tk.Frame(parent, bg=self.DUCK_YELLOW)
        tk.Label(frame, text="Ground truth:", bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 12, "bold")).pack(side="left")
        self.gt_canvas = tk.Canvas(frame, width=self._sw_w, height=self._sw_h,
                                   bg=self.DUCK_YELLOW, highlightthickness=0,
                                   cursor="hand2")
        self.gt_canvas.pack(side="left", padx=(8, 8))
        self.gt_canvas.bind("<Button-1>", lambda _e: self._toggle_gt())
        self.gt_text = tk.Label(frame, text="EKF", bg=self.DUCK_YELLOW,
                                fg=self.DARK, font=("DejaVu Sans", 10))
        self.gt_text.pack(side="left")
        self._draw_pill(self.gt_canvas, self.use_ground_truth)
        return frame

    def _toggle_gt(self):
        # Switch the planner's localization source: EKF estimate <-> ground truth.
        self.use_ground_truth = not self.use_ground_truth
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = self.use_ground_truth
        self.pub_use_gt.publish(msg)
        self._draw_pill(self.gt_canvas, self.use_ground_truth)
        self.gt_text.config(text="GT" if self.use_ground_truth else "EKF")
        self._set_status("Localization: GROUND TRUTH." if self.use_ground_truth
                         else "Localization: EKF estimate.", "blue")

    def _toggle_debug(self):
        # Locally driven (this GUI owns the planner-debug state).
        self.planner_debug = not self.planner_debug
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = self.planner_debug
        self.pub_debug_mode.publish(msg)
        self._draw_pill(self.debug_canvas, self.planner_debug)
        self.debug_text.config(text="on" if self.planner_debug else "off")
        if self.planner_debug:
            self.int_go_btn.grid()
        else:
            self.int_go_btn.grid_remove()
        self._set_status(
            "Planner debug ON — intersection decisions wait for GO."
            if self.planner_debug else "Planner debug OFF.", "blue")

    def _send_intersection_go(self):
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = True
        self.pub_debug_go.publish(msg)
        self._set_status("Intersection decision sent.", "green")

    def _cb_fsm_state(self, msg):
        # Drive the Mode switch from the real FSM state. Anything that isn't the
        # global joystick-override state counts as "autonomous / on".
        autonomous = (msg.state != "NORMAL_JOYSTICK_CONTROL")
        state = msg.state
        self.root.after(0, lambda: self._apply_switch_state(autonomous, state))

    def _apply_switch_state(self, autonomous, state):
        self.autonomous = autonomous
        self._draw_pill(self.mode_canvas, autonomous)
        self.mode_text.config(text=state if autonomous else "Joystick")

    def _set_status(self, text, color="black"):
        # Marshal onto the tkinter main loop — called from ROS threads.
        fg = self.STATUS_COLORS.get(color, color)
        self.root.after(0, lambda: (self.status_var.set(text),
                                    self.status_lbl.config(foreground=fg)))

    # ---- actions ----

    def _send(self):
        target = self.target_var.get().strip()
        if not target:
            self._set_status("Enter a target first.", "red")
            return
        self._set_status(f"Sending '{target}' …", "blue")
        # Service call can block; keep the UI responsive.
        threading.Thread(target=self._call_service, args=(target,),
                         daemon=True).start()

    def _call_service(self, target):
        try:
            rospy.wait_for_service(self.srv_name, timeout=5.0)
            proxy = rospy.ServiceProxy(self.srv_name, SetFSMState)
            # Sent as a literal string via ServiceProxy — no YAML parsing, so a
            # bare number like "2" goes through as-is (unlike `rosservice call`).
            proxy(target)
            kind = "node ID" if target.lstrip("-").isdigit() else "target"
            self._set_status(f"Sent {kind}: '{target}' — en route.", "green")
        except rospy.ROSException:
            self._set_status(f"Service not available:\n{self.srv_name}", "red")
        except Exception as e:
            self._set_status(f"Failed: {e}", "red")

    def _cb_arrived(self, msg):
        if msg.data:
            self._set_status("Arrived at target!", "green")

    def _cb_debug_image(self, msg):
        try:
            self._img_pil_raw = Image.open(io.BytesIO(bytes(msg.data)))
            self.root.after(0, self._redraw_debug_canvas)
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"[target_gui] Image decode failed: {e}")

    def _on_img_canvas_resize(self, _event):
        # Debounce: wait 100 ms after the last resize before redrawing.
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._redraw_debug_canvas)

    def _redraw_debug_canvas(self):
        self._resize_job = None
        if self._img_pil_raw is None:
            return
        w = self._img_canvas.winfo_width()
        h = self._img_canvas.winfo_height()
        if w < 2 or h < 2:
            return
        img = self._img_pil_raw.copy()
        img.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._img_photo = photo  # hold reference — GC would blank the canvas
        self._img_canvas.delete("all")
        self._img_canvas.create_image(w // 2, h // 2, anchor="center", image=photo)

    def _cb_camera_image(self, msg):
        try:
            img = Image.open(io.BytesIO(bytes(msg.data)))
            # Scale to the panel width, preserving aspect ratio. BILINEAR keeps
            # the camera view readable (NEAREST is only good for the tiny
            # upscaled debug images).
            scale = 560 / img.width
            img = img.resize((560, max(1, int(img.height * scale))), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self.root.after(0, lambda p=photo: self._update_cam_canvas(p))
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"[target_gui] Camera image decode failed: {e}")

    def _update_cam_canvas(self, photo):
        self._cam_photo = photo          # keep a ref or Tk garbage-collects it
        self._cam_canvas.config(width=photo.width(), height=photo.height())
        self._cam_canvas.delete("all")
        self._cam_canvas.create_image(0, 0, anchor="nw", image=photo)

    def _cb_segments_image(self, msg):
        try:
            img = Image.open(io.BytesIO(bytes(msg.data)))
            # always scale to fill the panel width (upscale small debug images)
            scale = 560 / img.width
            img = img.resize((560, int(img.height * scale)), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.root.after(0, lambda p=photo: self._update_seg_canvas(p))
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"[target_gui] Segment image decode failed: {e}")

    def _update_seg_canvas(self, photo):
        self._seg_photo = photo
        self._seg_canvas.config(width=photo.width(), height=photo.height())
        self._seg_canvas.delete("all")
        self._seg_canvas.create_image(0, 0, anchor="nw", image=photo)

    def _cb_clusters_image(self, msg):
        try:
            img = Image.open(io.BytesIO(bytes(msg.data)))
            scale = 560 / img.width
            img = img.resize((560, max(1, int(img.height * scale))), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.root.after(0, lambda p=photo: self._update_cluster_canvas(p))
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"[target_gui] Cluster image decode failed: {e}")

    def _update_cluster_canvas(self, photo):
        self._cluster_photo = photo
        self._cluster_canvas.config(width=photo.width(), height=photo.height())
        self._cluster_canvas.delete("all")
        self._cluster_canvas.create_image(0, 0, anchor="nw", image=photo)

    # ---- lifecycle ----

    def _check_ros(self):
        if rospy.is_shutdown():
            self.root.destroy()
        else:
            self.root.after(200, self._check_ros)

    def _on_close(self):
        rospy.signal_shutdown("GUI closed")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    TargetGUINode().run()
