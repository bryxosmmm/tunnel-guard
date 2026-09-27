#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /opt/tunnel-guard/install/setup.bash
set -u
exec "$@"
