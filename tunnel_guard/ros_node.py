"""Compatibility entry point for the single supported ROS 2 adapter.

The release node is :mod:`tunnel_guard.ros_node_buyanov`, launched through the
``tunnel_guard_ros`` ament package. Keeping this import-only module prevents
old ``python -m tunnel_guard.ros_node`` commands from silently selecting the
former, incompatible ``/perception/...`` topic contract.
"""

from .ros_node_buyanov import TunnelGuardNode, main

__all__ = ["TunnelGuardNode", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()
