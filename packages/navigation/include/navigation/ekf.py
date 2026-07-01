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
        # The node passes Q as 3x3 with entries [Q_xx, 0, 0; 0, Q_yy, 0; 0, 0, Q_tt],
        # but our noise vector is only [dX, dT] (2D), so W is 3x2 and we need a 2x2 Q.
        # Extract Q_xx = Q[0,0] and Q_tt = Q[2,2].
        if Q.shape == (3, 3):
            self.Q = np.array([
                [Q[0, 0], 0.0     ],
                [0.0,     Q[2, 2] ],
            ])
        else:
            self.Q = Q  # already 2x2
        self.R = R
        self.q_mutex = Lock()

    def predict(self, dX, dT):
        #print("predict")
        with self.q_mutex:
            theta = self.q[2]

            # Step 1: Propagate state using the unicycle kinematic model
            self.q[0] = self.q[0] + dX * np.cos(theta)
            self.q[1] = self.q[1] + dX * np.sin(theta)
            self.q[2] = self.q[2] + dT
            self.q[2] = wrap_angle(self.q[2])

            # Step 2: Jacobians of the process model
            #
            # F = df/dq  (3x3) — how the state maps to itself
            #     d/dx [ x + dX*cos(θ) ]   →  row 0: [1,  0,  -dX*sin(θ)]
            #     d/dy [ y + dX*sin(θ) ]   →  row 1: [0,  1,   dX*cos(θ)]
            #     d/dθ [ θ + dT        ]   →  row 2: [0,  0,   1         ]
            F = np.array([
                [1.0,  0.0, -dX * np.sin(theta)],
                [0.0,  1.0,  dX * np.cos(theta)],
                [0.0,  0.0,  1.0               ],
            ])

            # W = df/dw  (3x2) — how additive noise on [dX, dT] enters the state
            #     d/d(dX) [x + dX*cos(θ)]  →  col 0: [cos(θ), sin(θ), 0]
            #     d/d(dT) [θ + dT]         →  col 1: [0,      0,      1]
            W = np.array([
                [np.cos(theta), 0.0],
                [np.sin(theta), 0.0],
                [0.0,           1.0],
            ])

            # Step 3: Propagate the covariance
            #   P = F P Fᵀ + W Q Wᵀ
            self.P = F @ self.P @ F.T + W @ self.Q @ W.T

    def update(self, z: np.ndarray, tag_xy: np.ndarray):
        #print("update")

        """
        z      : [range, bearing]  — measured range (m) and bearing (rad) to the tag
        tag_xy : [tag_x, tag_y]   — known tag position in world frame
        """
        with self.q_mutex:
            tag_x, tag_y = tag_xy[0], tag_xy[1]
            x, y, theta = self.q[0], self.q[1], self.q[2]

            # Step 1: Predicted measurement from current state estimate
            dx = tag_x - x
            dy = tag_y - y
            r = np.sqrt(dx**2 + dy**2)

            # Guard against a tag sitting exactly on the robot
            if r < 1e-6:
                return

            rng_pred     = r
            bearing_pred = wrap_angle(np.arctan2(dy, dx) - theta)
            z_pred       = np.array([rng_pred, bearing_pred])

            # Step 2: Innovation (wrap the bearing component)
            y_innov    = z - z_pred
            y_innov[1] = wrap_angle(y_innov[1])

            # Step 3: Measurement Jacobian  H = dh/dq  (2x3)
            #
            # h = [ sqrt((lx-x)²+(ly-y)²),  atan2(ly-y, lx-x) - θ ]
            #
            # ∂r/∂x = -dx/r,   ∂r/∂y = -dy/r,   ∂r/∂θ = 0
            # ∂φ/∂x =  dy/r²,  ∂φ/∂y = -dx/r²,  ∂φ/∂θ = -1
            H = np.array([
                [-dx / r,   -dy / r,   0.0],
                [ dy / r**2, -dx / r**2, -1.0],
            ])

            # Step 4: Kalman gain
            #   K = P Hᵀ (H P Hᵀ + R)⁻¹
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)

            # Step 5: Update state and covariance
            self.q   = self.q + K @ y_innov
            self.q[2] = wrap_angle(self.q[2])

            # Joseph form for numerical stability: P = (I - KH) P (I - KH)ᵀ + K R Kᵀ
            I_KH    = np.eye(3) - K @ H
            self.P  = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T