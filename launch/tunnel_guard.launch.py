"""Launch the adapter and optional RViz using the installed Python environment."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config", default_value="/opt/tunnel-guard/configs/detector.json"
            ),
            DeclareLaunchArgument("input_topic", default_value="/lidar_points"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("input_reliability", default_value="reliable"),
            DeclareLaunchArgument("input_timeout_s", default_value="3.0"),
            ExecuteProcess(
                cmd=[
                    "python3",
                    "-m",
                    "tunnel_guard.ros_node",
                    "--ros-args",
                    "-p",
                    ["config:=", LaunchConfiguration("config")],
                    "-p",
                    ["input_topic:=", LaunchConfiguration("input_topic")],
                    "-p",
                    ["input_reliability:=", LaunchConfiguration("input_reliability")],
                    "-p",
                    ["input_timeout_s:=", LaunchConfiguration("input_timeout_s")],
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=["rviz2", "-d", "/opt/tunnel-guard/rviz/tunnel_guard.rviz"],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )
