"""Sample the M3 static-bag timing and LiDAR sensitivity envelope."""
from dataclasses import replace

import numpy as np

from .m3_warehouse import DynamicsProfile


def measured_variation(rng: np.random.Generator) -> DynamicsProfile:
    """Use the four static Gazebo runs in experiments/m3/fortress-motion.json.

    Keep drive limits nominal. The source motion intervals combine controller
    ramps, simulated physics and sampling, so they are not actuator limits.
    """
    return replace(
        DynamicsProfile(),
        drive_delay_ticks=int(rng.integers(1, 3)),
        scan_noise_std=float(rng.uniform(0.0297, 0.0302)),
    )
