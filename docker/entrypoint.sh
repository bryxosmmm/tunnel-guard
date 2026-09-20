#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
source /opt/tunnel-guard/install/setup.bash
exec "$@"
