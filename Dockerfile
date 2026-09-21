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
        ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tunnel-guard
COPY pyproject.toml uv.lock README.md setup.py MANIFEST.in ./
COPY cpp ./cpp
COPY docker/constraints.txt ./docker/constraints.txt
COPY tunnel_guard ./tunnel_guard
COPY configs ./configs
COPY docs ./docs
COPY rviz ./rviz
COPY ros2_ws ./ros2_ws
COPY launch ./launch

RUN python3 -m pip install --no-cache-dir --upgrade pip==24.3.1 setuptools==75.8.0 wheel==0.45.1 \
    && python3 -m pip install -r docker/constraints.txt packaging==24.2 \
    && python3 -m pip install --no-build-isolation --no-deps . \
    && python3 -c "from tunnel_guard import _native; import tunnel_guard.ros_node" \
    && source /opt/ros/humble/setup.bash \
    && colcon build --merge-install --base-paths ros2_ws/src

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "tunnel_guard_ros", "tunnel_guard.launch.py"]
