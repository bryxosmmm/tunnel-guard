FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        python3-colcon-common-extensions \
        python3-dev \
        python3-pip \
        ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tunnel-guard
COPY pyproject.toml uv.lock README.md ./
COPY tunnel_guard ./tunnel_guard
COPY configs ./configs
COPY docs ./docs
COPY rviz ./rviz
COPY ros2_ws ./ros2_ws

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir . \
    && source /opt/ros/humble/setup.bash \
    && colcon build --merge-install --base-paths ros2_ws/src

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "tunnel_guard_ros", "tunnel_guard.launch.py"]
