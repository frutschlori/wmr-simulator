__all__ = [
    "POLOLU_TRAJ_CONTROL_COLUMNS",
    "PololuReference",
    "PoseSplines",
    "ReferenceTrajectory",
    "append_bridge_reference",
    "drop_repeated_poses",
    "export_latest_reference",
    "export_pololu_reference_pickle",
    "export_robot_config",
    "fit_pose_splines",
    "format_pololu_reference",
    "format_robot_config",
    "load_imu_gyro_z",
    "load_latest_reference_trajectory",
    "load_pololu_traj_control_log",
    "load_pololu_reference",
    "load_reference_trajectory",
    "load_robot_config_file",
    "reference_states_from_pololu_reference",
    "robot_config_values",
]


def __getattr__(name):
    if name in {
        "POLOLU_TRAJ_CONTROL_COLUMNS",
        "load_imu_gyro_z",
        "load_pololu_traj_control_log",
    }:
        from wmr_simulator.pololu import log_loader

        return getattr(log_loader, name)
    if name in {
        "PololuReference",
        "export_pololu_reference_pickle",
        "load_pololu_reference",
        "reference_states_from_pololu_reference",
    }:
        from wmr_simulator.pololu import reference_importer

        return getattr(reference_importer, name)
    if name in {
        "ReferenceTrajectory",
        "export_latest_reference",
        "format_pololu_reference",
        "load_latest_reference_trajectory",
        "load_reference_trajectory",
    }:
        from wmr_simulator.pololu import reference_exporter

        return getattr(reference_exporter, name)
    if name in {
        "PoseSplines",
        "drop_repeated_poses",
        "fit_pose_splines",
    }:
        from wmr_simulator.pololu import pose_smoothing

        return getattr(pose_smoothing, name)
    if name == "append_bridge_reference":
        from wmr_simulator.pololu.bridge_exporter import append_bridge_reference

        return append_bridge_reference
    if name in {
        "export_robot_config",
        "format_robot_config",
        "load_robot_config_file",
        "robot_config_values",
    }:
        from wmr_simulator.pololu import robot_config

        return getattr(robot_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
