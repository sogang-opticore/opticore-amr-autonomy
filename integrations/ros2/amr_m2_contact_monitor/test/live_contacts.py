"""Calibrate the Fortress plugin with clean, robot-into-rack and actor-into-robot cases."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid


def run(plugin_dir):
    env = os.environ.copy()
    env["IGN_PARTITION"] = "m2_calibration_" + uuid.uuid4().hex
    env["IGN_GAZEBO_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(
        p for p in (str(plugin_dir), env.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH")) if p
    )
    fixture = Path(__file__).with_name("contact_fixture.sdf").resolve()
    samples = queue.Queue()

    def ingest(stream):
        for line in stream:
            label, sep, value = line.strip().partition(":")
            if sep and label == "data":
                # ign topic -e prints StringMsg in protobuf text format.
                samples.put(json.loads(json.loads(value.strip())))

    def expect(predicate, description, minimum_stamp=-1):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            try:
                item = samples.get(timeout=0.25)
            except queue.Empty:
                continue
            if item["stamp"] > minimum_stamp and predicate(item):
                return item["stamp"]
        raise RuntimeError("Missing physics evidence: " + description)

    def command(service, reqtype, request):
        result = subprocess.run(
            ["ign", "service", "-s", "/world/m2_contact_fixture/" + service,
             "--reqtype", reqtype, "--reptype", "ignition.msgs.Boolean",
             "--timeout", "3000", "--req", request], env=env, check=True,
            capture_output=True, text=True, timeout=8
        )
        if "data: true" not in result.stdout:
            raise RuntimeError("Gazebo rejected fixture command: " + result.stdout)

    def move(model, x, z):
        request = f'name: "{model}", position: {{x:{x},y:0,z:{z}}}, orientation: {{w:1}}'
        command("set_pose", "ignition.msgs.Pose", request)

    def observed_xy(item, x):
        xy = item.get("actor_world_xy")
        return (item["actor_pose_valid"] and isinstance(xy, list)
                and abs(xy[0] - x) < 0.001 and abs(xy[1]) < 0.001)

    def clear(item):
        return item["source"] == "gazebo_contacts" and item["collision"] is False

    def hit(model):
        return lambda item: item["collision"] is True and ["amr1", model] in item["pairs"]

    processes = []
    with tempfile.TemporaryFile(mode="w+") as log:
        try:
            processes.append(subprocess.Popen(
                ["ign", "gazebo", "-s", "-r", str(fixture)], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            ))
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                listing = subprocess.run(["ign", "topic", "-l"], env=env,
                                         capture_output=True, text=True, timeout=5)
                if "/amr1/m2/contacts" in listing.stdout.splitlines():
                    break
                if processes[0].poll() is not None:
                    raise RuntimeError("Fortress exited before loading the contact plugin")
                time.sleep(0.15)
            else:
                raise RuntimeError("Contact topic did not appear")
            # A pipe normally block-buffers ign topic; line buffering preserves short contacts.
            listener = subprocess.Popen(
                ["stdbuf", "-oL", "ign", "topic", "-e", "-t", "/amr1/m2/contacts"],
                env=env, stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True
            )
            processes.append(listener)
            threading.Thread(target=ingest, args=(listener.stdout,), daemon=True).start()
            ready = expect(lambda item: clear(item) and observed_xy(item, -2.0),
                           "grounded wheels and observed actor start")
            move("amr1", 1.65, 0.11)
            wall_stamp = expect(hit("rack_fixture"), "chassis against fixed rack", ready)
            move("amr1", 0, 0.11)
            clear_stamp = expect(clear, "clear after reset", wall_stamp)
            move("m2_mission_obstacle", 0.34, 0.30)
            impact_stamp = expect(lambda item: hit("m2_mission_obstacle")(item) and observed_xy(item, 0.34),
                                  "actor moved into stationary AMR", clear_stamp)
            command("remove", "ignition.msgs.Entity", 'name: "m2_mission_obstacle", type: MODEL')
            expect(lambda item: clear(item) and item["actor_pose_valid"] and item["actor_world_xy"] is None,
                   "actor physically removed", impact_stamp)
        except Exception:
            log.seek(0)
            sys.stderr.write(log.read()[-4000:])
            raise
        finally:
            for proc in reversed(processes):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
