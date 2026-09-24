FROM ros:humble-ros-base-jammy
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential libgl1 libgomp1 ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/tunnel-guard
COPY docker/constraints.txt ./docker/constraints.txt
RUN python3 -m pip install --upgrade pip==24.3.1 setuptools==75.8.0 wheel==0.45.1 \
    && python3 -m pip install -r docker/constraints.txt
COPY pyproject.toml setup.py MANIFEST.in ./
COPY cpp ./cpp
COPY tunnel_guard ./tunnel_guard
RUN python3 -m pip install --upgrade packaging==24.2 \
    && python3 -m pip install --no-build-isolation --no-deps . \
    && python3 -c "from tunnel_guard import _native" \
    && python3 -c "import tunnel_guard.ros_node"
COPY configs ./configs
COPY rviz ./rviz
COPY launch ./launch
CMD ["python3", "-m", "tunnel_guard.ros_node"]
