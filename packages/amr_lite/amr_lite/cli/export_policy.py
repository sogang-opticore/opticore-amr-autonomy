from __future__ import annotations

import argparse
import json

from amr_lite.deploy.export import export_bundle
from amr_lite.deploy.runtime import PolicyRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint as a torch-free policy bundle")
    parser.add_argument("--checkpoint", required=True, help="BC, M1 PPO or PPO v2 checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="bundle directory")
    parser.add_argument("--robot-profile", help="profile for checkpoints that do not record one "
                                                "(BC and M1 PPO default to lite_default)")
    parser.add_argument("--selection", help="JSON file with the selection evidence to record")
    args = parser.parse_args()
    selection = json.load(open(args.selection, encoding="utf-8")) if args.selection else None
    manifest = export_bundle(args.checkpoint, args.output, robot_profile=args.robot_profile, selection=selection)
    error = PolicyRuntime.load(args.output).verify_golden()
    print(json.dumps({"bundle": args.output, "policy": manifest["policy"]["kind"],
                      "mode": manifest["policy"]["mode"], "robot_profile": manifest["robot_profile"],
                      "golden_max_error": error, "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
