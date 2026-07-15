#!/usr/bin/env python3
import numpy as np
from multiprocessing import Lock


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    def __init__(self, q_0: np.ndarray, P_0: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.q = q_0
        self.P = P_0
        self.Q = Q
        self.R = R
        self.q_mutex = Lock()

    def predict(self, dX, dT):
        with self.q_mutex:
            theta = self.q[2]
            phi = theta + 0.5 * dT

            self.q[0] = self.q[0] + dX * np.cos(phi)
            self.q[1] = self.q[1] + dX * np.sin(phi)
            self.q[2] = wrap_angle(self.q[2] + dT)

            F = np.array([
                [1.0, 0.0, -dX * np.sin(phi)],
                [0.0, 1.0,  dX * np.cos(phi)],
                [0.0, 0.0,  1.0],
            ])

            W = np.array([
                [np.cos(phi), -0.5 * dX * np.sin(phi)],
                [np.sin(phi),  0.5 * dX * np.cos(phi)],
                [0.0,          1.0],
            ])

            self.P = F @ self.P @ F.T + W @ self.Q @ W.T

    def update(self, z: np.ndarray, tag_xy: np.ndarray):
        with self.q_mutex:
            tag_x, tag_y = tag_xy[0], tag_xy[1]
            x, y, theta = self.q[0], self.q[1], self.q[2]

            dx = tag_x - x
            dy = tag_y - y
            r = np.sqrt(dx**2 + dy**2)

            if r < 1e-6:
                return None, None

            z_pred = np.array([r, wrap_angle(np.arctan2(dy, dx) - theta)])

            y_innov = np.asarray(z, dtype=float) - z_pred
            y_innov[1] = wrap_angle(y_innov[1])

            H = np.array([
                [-dx / r,    -dy / r,     0.0],
                [ dy / r**2, -dx / r**2, -1.0],
            ])

            S = H @ self.P @ H.T + self.R
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv

            self.q = self.q + K @ y_innov
            self.q[2] = wrap_angle(self.q[2])

            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

            return y_innov, S
