# WarehouseEnv is resolved lazily so that light modules (dynamics, raycast,
# maps) can be imported by amr_lite.sensors and the ROS adapter without
# pulling in the full environment and creating an import cycle.
__all__ = ["WarehouseEnv"]


def __getattr__(name: str):
    if name == "WarehouseEnv":
        from .warehouse_env import WarehouseEnv
        return WarehouseEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
