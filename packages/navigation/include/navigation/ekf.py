#!/usr/bin/env python3
import numpy as np
from multiprocessing import Lock


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    def __init__(self, q_0: np.ndarray, P_0: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.q = q_0
        self.P = P_0
        if Q.shape == (3, 3):
            self.Q = np.array([[Q[0, 0], 0.0], [0.0, Q[2, 2]]])
        else:
            self.Q = Q
        self.R = R
        self.q_mutex = Lock()
        self.history = []  # one entry per predict() call, for RTS smoothing

        # Add initial state to history for proper smoothing
        self.history.append({
            'q_prior': q_0.copy(),  # No prior before initial
            'F':       np.eye(3),   # No motion before initial
            'q_pred':  q_0.copy(),
            'P_pred':  P_0.copy(),
            'q_filt':  q_0.copy(),
            'P_filt':  P_0.copy(),
        })
    def predict(self, dX, dT):
        with self.q_mutex:
            theta = self.q[2]
            q_prior = self.q.copy()

            self.q[0] = self.q[0] + dX * np.cos(theta)
            self.q[1] = self.q[1] + dX * np.sin(theta)
            self.q[2] = self.q[2] + dT
            self.q[2] = wrap_angle(self.q[2])

            F = np.array([
                [1.0, 0.0, -dX * np.sin(theta)],
                [0.0, 1.0,  dX * np.cos(theta)],
                [0.0, 0.0,  1.0               ],
            ])
            W = np.array([
                [np.cos(theta), 0.0],
                [np.sin(theta), 0.0],
                [0.0,           1.0],
            ])

            self.P = F @ self.P @ F.T + W @ self.Q @ W.T

            # Placeholder filt = pred; overwritten by finalize_step() if an
            # update() happens this cycle. If no update happens, this is
            # already the correct "filtered" value (no measurement to fuse).
            self.history.append({
                'q_prior': q_prior,
                'F':       F,
                'q_pred':  self.q.copy(),
                'P_pred':  self.P.copy(),
                'q_filt':  self.q.copy(),
                'P_filt':  self.P.copy(),
            })

    def update(self, z: np.ndarray, tag_xy: np.ndarray):
        # ... unchanged from your version ...
        with self.q_mutex:
            tag_x, tag_y = tag_xy[0], tag_xy[1]
            x, y, theta = self.q[0], self.q[1], self.q[2]
            dx, dy = tag_x - x, tag_y - y
            r = np.sqrt(dx**2 + dy**2)
            if r < 1e-6:
                return
            z_pred = np.array([r, wrap_angle(np.arctan2(dy, dx) - theta)])
            y_innov = z - z_pred
            y_innov[1] = wrap_angle(y_innov[1])
            H = np.array([[-dx / r, -dy / r, 0.0],
                          [dy / r**2, -dx / r**2, -1.0]])
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.q = self.q + K @ y_innov
            self.q[2] = wrap_angle(self.q[2])
            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def finalize_step(self):
        """Call once per predict() cycle, after any update() calls, so the
        most recent history entry reflects the post-measurement state."""
        with self.q_mutex:
            if not self.history:
                return
            self.history[-1]['q_filt'] = self.q.copy()
            self.history[-1]['P_filt'] = self.P.copy()

    def rts_smooth(self):
        """
        Extended RTS smoother backward pass. Returns an (N,3) array of
        smoothed [x, y, theta], oldest to newest, over the stored history.
        """
        print("rts_smooth")
        with self.q_mutex:
            n = len(self.history)
            if n == 0:
                return np.zeros((0, 3))

            q_smooth = [None] * n
            P_smooth = [None] * n
            
            # Start from the last state (same as before)
            q_smooth[-1] = self.history[-1]['q_filt'].copy()
            P_smooth[-1] = self.history[-1]['P_filt'].copy()

            # Backward pass - but skip the first state (k=0)
            for k in range(n - 2, -1, -1):  # Changed: stop at 1 instead of 0
                F_next      = self.history[k + 1]['F']
                P_pred_next = self.history[k + 1]['P_pred']
                q_pred_next = self.history[k + 1]['q_pred']
                P_filt_k    = self.history[k]['P_filt']
                q_filt_k    = self.history[k]['q_filt']

                # Regularize
                P_pred_reg = P_pred_next + np.eye(3) * 1e-9
                try:
                    P_pred_inv = np.linalg.inv(P_pred_reg)
                except np.linalg.LinAlgError:
                    P_pred_inv = np.linalg.pinv(P_pred_reg)

                C_k = P_filt_k @ F_next.T @ P_pred_inv

                innov = q_smooth[k + 1] - q_pred_next
                innov[2] = wrap_angle(innov[2])

                q_smooth[k] = q_filt_k + C_k @ innov
                q_smooth[k][2] = wrap_angle(q_smooth[k][2])
                P_smooth[k] = P_filt_k + C_k @ (P_smooth[k + 1] - P_pred_next) @ C_k.T

            return np.array(q_smooth)