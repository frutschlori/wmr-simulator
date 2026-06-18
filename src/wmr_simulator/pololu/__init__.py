__all__ = [
    "POLOLU_TRAJ_CONTROL_COLUMNS",
    "PololuReference",
    "ReferenceTrajectory",
    "export_latest_reference",
    "export_pololu_reference_pickle",
    "format_pololu_reference",
    "load_latest_reference_trajectory",
    "load_pololu_traj_control_log",
    "load_pololu_reference",
    "load_reference_trajectory",
    "reference_states_from_pololu_reference",
]


def __getattr__(name):
    if name in {
        "POLOLU_TRAJ_CONTROL_COLUMNS",
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

        return getattr(reference_formatter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
