from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> float:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
    """

    dticks = ticks - prev_ticks
    alpha = 2*np.pi/resolution
    dphi = dticks*alpha
    # ---
    return dphi


def get_odometry(
    R: float,
    baseline: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float]:

    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        dA:                  forward displacement
        dtheta:              rotation
    """

    d_left = R * delta_phi_left
    d_right = R * delta_phi_right

    dA = (d_left + d_right) / 2

    dtheta = (d_right - d_left) / baseline

    return dA, dtheta
