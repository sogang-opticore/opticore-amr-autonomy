#!/usr/bin/env bash
# Run a committed development revision with the existing Humble/Fortress CPU image.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd -P)
suite=${1:-all}
[[ $# -le 1 ]] || { echo "Expected at most one suite name." >&2; exit 2; }
case "$suite" in paired|all|faults|pair|fleet|transport|sequence) ;; *) echo "Usage: $0 [paired|all|faults|pair|fleet|transport|sequence]" >&2; exit 2;; esac
kit=${M5_PILOT_KIT:-"$repo/.cowork/m5/input/kit.tgz"}
candidates=${M5_RUNS:-"$repo/.cowork/m4-mixed/runs"}
image=${M5_IMAGE:-opticore-m2-local:humble-fortress}
test -r "$kit" || { echo "Set M5_PILOT_KIT to the M2 661ebc8 pilot archive." >&2; exit 2; }
kit=$(cd "$(dirname "$kit")" && pwd -P)/$(basename "$kit")
test "$(shasum -a 256 "$kit" | cut -d' ' -f1)" = 97e3cc308e0ffb9dd24bfd856bcff7a6ac8c2295fec34012e577ddab351e3aed || {
  echo "Pilot archive checksum differs from the pinned M2 archive." >&2; exit 2;
}
docker info --format 'Engine: {{.OperatingSystem}} / {{.Architecture}}'
image_id=$(docker image inspect "$image" --format '{{.Id}}')
runs_dir="$repo/.cowork/m5"
mkdir -p "$runs_dir"
run=$(mktemp -d "$runs_dir/live-XXXXXXXX")
mkdir "$run/input" "$run/output"
if [[ $suite == all || $suite == fleet ]]; then
  if [[ -n ${M5_UPSTREAM_REPO:-} ]]; then
    git -C "$M5_UPSTREAM_REPO" archive --format=tar --output="$run/input/fleet.tar" \
      e8c9197ac2801c2770c83a7ee696b2b992d473b8 ros2_ws/src/amr_fleet
  else
    cp "${M5_FLEET_ARCHIVE:-$runs_dir/input/fleet.tar}" "$run/input/fleet.tar" || {
      echo "Set M5_UPSTREAM_REPO to the opticore-amr checkout for the fleet manager trial." >&2; exit 2;
    }
    test "$(shasum -a 256 "$run/input/fleet.tar" | cut -d' ' -f1)" = 450e0b9a8615df8dc6f8003a0e1713786176a11af07404fb7d89091f731561ed || {
      echo "Fleet package archive checksum mismatch." >&2; exit 2;
    }
  fi
fi
git -C "$repo" bundle create "$run/input/autonomy.bundle" HEAD
git -C "$repo" rev-parse HEAD > "$run/input/revision"
for seed in 7 19 31; do
  mkdir -p "$run/input/candidates/$seed"
  cp -R "$candidates/$seed/bundle" "$run/input/candidates/$seed/"
done
echo "$image_id" > "$run/output/docker-image-id"
echo "Results: $run/output"
docker run --rm --init --shm-size=1g \
  --mount "type=bind,source=$kit,target=/pilot.tgz,readonly" \
  --mount "type=bind,source=$run/input,target=/input,readonly" \
  --mount "type=bind,source=$run/output,target=/results" \
  -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe \
  "$image_id" bash -ec '
    cd /tmp
    tar -xzf /pilot.tgz
    (cd m2-pilot && sha256sum --quiet -c SHA256SUMS)
    if test -e /input/fleet.tar; then tar -xf /input/fleet.tar -C m2-pilot/upstream; fi
    git init -q autonomy
    cd autonomy
    git fetch -q /input/autonomy.bundle HEAD
    git checkout -q --detach FETCH_HEAD
    test "$(git rev-parse HEAD)" = "$(cat /input/revision)"
    exec xvfb-run -a -s "-screen 0 1280x720x24" bash scripts/m5/run_on_humble.sh /tmp/m2-pilot/upstream /input/candidates /results/trials "$1"
  ' m5 "$suite" 2>&1 | tee "$run/output/session.log"
