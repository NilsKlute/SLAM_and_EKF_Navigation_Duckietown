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
import os
import threading
import tkinter as tk
from tkinter import ttk

import yaml
import rospy
from duckietown_msgs.srv import SetFSMState
from duckietown_msgs.msg import BoolStamped


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
        label_map_file = rospy.get_param("~label_map_file", "")

        self.srv_name = f"/{self.veh}/target_selection_node/select_target"
        self.targets = self._load_targets(label_map_file)

        # Live "arrived" feedback from the planner (latched topic).
        rospy.Subscriber(
            f"/{self.veh}/graph_planner_node/arrived_at_target",
            BoolStamped, self._cb_arrived, queue_size=1)

        self._build_ui()

    # ---- target list ----

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
        self.root.minsize(340, 175)
        self.root.configure(bg=self.DUCK_YELLOW)

        pad = dict(padx=14, pady=8)

        tk.Label(self.root, text="Target location:",
                 bg=self.DUCK_YELLOW, fg=self.DARK,
                 font=("DejaVu Sans", 12, "bold")).grid(
                     row=0, column=0, columnspan=2, sticky="w", **pad)

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
        self.root.option_add("*TCombobox*Listbox.font", "DejaVu\\ Sans 11")

        self.combo = ttk.Combobox(self.root, textvariable=self.target_var,
                                  values=self.targets, width=22,
                                  style="Duck.TCombobox",
                                  font=("DejaVu Sans", 11))
        self.combo.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)
        self.combo.bind("<Return>", lambda _e: self._send())

        self.go_btn = tk.Button(self.root, text="GO", command=self._send,
                                bg=self.DUCK_BLUE, fg=self.WHITE,
                                activebackground=self.DUCK_BLUE_ACTIVE,
                                activeforeground=self.WHITE,
                                font=("DejaVu Sans", 13, "bold"),
                                relief="flat", bd=0, padx=10, pady=8,
                                cursor="hand2")
        self.go_btn.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        self.status_var = tk.StringVar(value=f"Service: {self.srv_name}")
        self.status_lbl = tk.Label(self.root, textvariable=self.status_var,
                                   bg=self.DUCK_YELLOW, fg=self.DARK,
                                   wraplength=310, justify="left",
                                   font=("DejaVu Sans", 9))
        self.status_lbl.grid(row=3, column=0, columnspan=2, sticky="w", **pad)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Shut the GUI down cleanly when ROS dies (e.g. Ctrl-C in the terminal).
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._check_ros)

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
            proxy(target)
            self._set_status(f"Target set: '{target}' — en route.", "green")
        except rospy.ROSException:
            self._set_status(f"Service not available:\n{self.srv_name}", "red")
        except Exception as e:
            self._set_status(f"Failed: {e}", "red")

    def _cb_arrived(self, msg):
        if msg.data:
            self._set_status("Arrived at target!", "green")

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
