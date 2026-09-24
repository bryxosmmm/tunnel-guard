# Delivery-path audit

This document formerly described a second live adapter under
`/perception/...`. That adapter and its identity-TF workaround are retired.
They were not a verified calibration and must not be used for release.

The current auditable delivery path is:

`Docker build → tunnel_guard_ros launch → ROS 2 PointCloud2 bag play → /tunnel_guard result, distance, cloud and markers`

Its exact commands, topic types, QoS, timestamp rules, watchdog behavior and
known evidence gaps are maintained in [ROS2.md](ROS2.md). The change is
deliberately narrow: it removes a contradictory interface rather than claiming
that this development machine has executed the target container or RViz GUI.
