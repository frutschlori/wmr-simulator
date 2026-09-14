"""Save/load residual dynamics models.

A checkpoint is a single pickle holding the model config (so the module
skeleton can be rebuilt), the array leaves (expert weights + gate + normalization
stats, as numpy), the nominal plant parameters the residual was trained against,
and free-form training metadata. Loading is explicit: rebuild the skeleton from
the config, then swap in the saved leaves.

A residual is a correction *of* one nominal model -- its targets are measured
minus that model's prediction -- so it is only meaningful added to that same
model. Loading therefore takes the robot block of the plant it is about to be
added to and refuses a checkpoint trained against different parameters.
"""

import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.residual_model.residual import ResidualEnsemble, init_residual_model

_CHECKPOINT_FORMAT = "wmr-residual-ensemble-v1"

# The nominal-plant parameters a residual's targets depend on (DiffDrive.step).
PLANT_PARAMETER_NAMES = ("wheel_radius", "base_diameter", "max_wheel_speed", "time_constant")
# float32 round trips (PhysicalParams) move the yaml values by ~1e-7 relative.
_PLANT_PARAMETER_RTOL = 1e-5


def plant_parameters(robot) -> dict[str, float]:
    """The nominal-plant parameters of a problem's robot block (a mapping) or of
    a ``PhysicalParams``."""
    values = robot._asdict() if hasattr(robot, "_asdict") else robot
    return {name: float(values[name]) for name in PLANT_PARAMETER_NAMES}


def save_residual_model(
    path: str | Path,
    model: ResidualEnsemble,
    config: dict,
    plant: dict,
    metadata: dict | None = None,
):
    """``config`` must hold the init_residual_model kwargs
    (input_dim, output_dim, num_experts, hidden_sizes, spectral_norm_cap);
    ``plant`` is the robot block (or ``PhysicalParams``) of the nominal model the
    residual was trained against."""
    params, _ = eqx.partition(model, eqx.is_array)
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(params)]
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "config": dict(config),
        "leaves": leaves,
        "plant_params": plant_parameters(plant),
        "metadata": dict(metadata) if metadata else {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def load_residual_model(path: str | Path, plant) -> tuple[ResidualEnsemble, dict]:
    """Returns (model, checkpoint) where checkpoint holds config and metadata.

    ``plant`` is the robot block (or ``PhysicalParams``) of the nominal model the
    residual is about to be added to; a checkpoint trained against any other
    parameters raises.
    """
    with open(path, "rb") as file:
        payload = pickle.load(file)
    if payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"Unrecognized residual model checkpoint format in {path}")
    if "plant_params" not in payload:
        raise ValueError(
            f"Residual checkpoint {path} does not record the nominal plant it was trained "
            "against; retrain it."
        )
    trained = payload["plant_params"]
    requested = plant_parameters(plant)
    mismatched = {
        name: (trained[name], requested[name])
        for name in PLANT_PARAMETER_NAMES
        if not np.isclose(trained[name], requested[name], rtol=_PLANT_PARAMETER_RTOL, atol=0.0)
    }
    if mismatched:
        details = ", ".join(f"{name} trained {a:.6g} vs plant {b:.6g}" for name, (a, b) in mismatched.items())
        raise ValueError(
            f"Residual checkpoint {path} was trained against a different nominal plant ({details}); "
            "a residual is only valid on the model it was trained on."
        )

    config = payload["config"]
    skeleton = init_residual_model(
        jax.random.PRNGKey(0),
        input_dim=int(config["input_dim"]),
        output_dim=int(config.get("output_dim", 2)),
        num_experts=int(config["num_experts"]),
        hidden_sizes=tuple(config["hidden_sizes"]),
        spectral_norm_cap=float(config.get("spectral_norm_cap", 0.0)),
    )
    params, static = eqx.partition(skeleton, eqx.is_array)
    treedef = jax.tree_util.tree_structure(params)
    leaves = [jnp.asarray(leaf, dtype=jnp.float32) for leaf in payload["leaves"]]
    if len(leaves) != treedef.num_leaves:
        raise ValueError(
            f"Checkpoint {path} has {len(leaves)} array leaves, expected {treedef.num_leaves};"
            " config does not match the saved parameters."
        )
    model = eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)
    return model, payload
