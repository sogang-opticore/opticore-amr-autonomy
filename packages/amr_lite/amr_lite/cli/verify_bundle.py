from __future__ import annotations

import argparse
import json
import sys

from amr_lite.config import resolve_env_config
from amr_lite.deploy.runtime import BundleError, PolicyRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a policy bundle without PyTorch")
    parser.add_argument("bundle")
    parser.add_argument("--robot-profile", default="opticore_gazebo",
                        help="profile the bundle must match for deployment")
    args = parser.parse_args()
    try:
        runtime = PolicyRuntime.load(args.bundle)
    except BundleError as error:
        print(json.dumps({"ok": False, "fault": error.code, "detail": str(error)}, indent=2))
        sys.exit(1)
    expected = resolve_env_config(args.robot_profile)
    v_max, w_max = runtime.action_limits
    mismatches = {key: (runtime.robot_profile["values"].get(key), expected.get(key))
                  for key in ("v_max", "w_max", "accel_max", "decel_max", "angular_accel_max", "lidar_model")
                  if runtime.robot_profile["values"].get(key) != expected.get(key)}
    report = {"ok": not mismatches, "golden_max_error": runtime.verify_golden(),
              "profile": runtime.robot_profile["name"], "profile_source": runtime.robot_profile["source"],
              "mode": runtime.mode, "action_limits": [v_max, w_max], "profile_mismatches": mismatches}
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
