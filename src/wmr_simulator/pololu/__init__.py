__all__ = [
    "POLOLU_TRAJ_CONTROL_COLUMNS",
    "PololuTrajControlLog",
    "ReferenceTrajectory",
    "export_latest_reference",
    "format_pololu_reference",
    "load_latest_reference_trajectory",
    "load_pololu_traj_control_log",
    "load_reference_trajectory",
]


def __getattr__(name):
    if name in {
        "POLOLU_TRAJ_CONTROL_COLUMNS",
        "PololuTrajControlLog",
        "load_pololu_traj_control_log",
    }:
        from wmr_simulator.pololu import log_loader

        return getattr(log_loader, name)
    if name in {
        "ReferenceTrajectory",
        "export_latest_reference",
        "format_pololu_reference",
        "load_latest_reference_trajectory",
        "load_reference_trajectory",
    }:
        from wmr_simulator.pololu import reference_formatter

        return getattr(reference_formatter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
