"""Launch the adapter and optional RViz using the installed Python environment."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value="/opt/tunnel-guard/configs/detector-native.json"),
            DeclareLaunchArgument("input_topic", default_value="/lidar_points"),
            DeclareLaunchArgument("input_reliability", default_value="reliable"),
            DeclareLaunchArgument("input_timeout_s", default_value="3.0"),
            Node(
                package="tunnel_guard_ros",
                executable="tunnel_guard_node",
                name="tunnel_guard",
                output="screen",
                parameters=[{
                    "config_path": LaunchConfiguration("config_path"),
                    "input_topic": LaunchConfiguration("input_topic"),
                    "input_reliability": LaunchConfiguration("input_reliability"),
                    "input_timeout_s": LaunchConfiguration("input_timeout_s"),
                }],
            ),
        ]
    )
