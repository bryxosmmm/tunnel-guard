FROM ghcr.io/astral-sh/uv:0.11.6 AS uv

FROM ros:humble-ros-base-jammy

COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_INSTALL_DIR=/app/.python \
    UV_CACHE_DIR=/app/.uv-cache \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python 3.12 --no-install-project

COPY tunnel_guard ./tunnel_guard
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --python 3.12 \
    && rm -rf /app/.uv-cache

ENTRYPOINT ["/app/.venv/bin/python", "-m", "tunnel_guard.run"]
CMD ["--help"]
