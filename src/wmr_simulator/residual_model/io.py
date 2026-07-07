"""Save/load residual dynamics models.

A checkpoint is a single pickle holding the model config (so the module
skeleton can be rebuilt), the array leaves (parameters + normalization stats,
as numpy), and free-form training metadata. Loading is explicit: rebuild the
skeleton from the config, then swap in the saved leaves.
"""

import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.residual_model.residual import ResidualDynamicsModel, init_residual_model

_CHECKPOINT_FORMAT = "wmr-residual-model-v1"


def save_residual_model(
    path: str | Path,
    model: ResidualDynamicsModel,
    config: dict,
    metadata: dict | None = None,
):
    """``config`` must hold the init_residual_model kwargs
    (input_dim, hidden_width, hidden_depth, output_dim)."""
    params, _ = eqx.partition(model, eqx.is_array)
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(params)]
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "config": dict(config),
        "leaves": leaves,
        "metadata": dict(metadata) if metadata else {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def load_residual_model(path: str | Path) -> tuple[ResidualDynamicsModel, dict]:
    """Returns (model, checkpoint) where checkpoint holds config and metadata."""
    with open(path, "rb") as file:
        payload = pickle.load(file)
    if payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"Unrecognized residual model checkpoint format in {path}")

    config = payload["config"]
    skeleton = init_residual_model(
        jax.random.PRNGKey(0),
        input_dim=int(config["input_dim"]),
        hidden_width=int(config["hidden_width"]),
        hidden_depth=int(config["hidden_depth"]),
        output_dim=int(config.get("output_dim", 3)),
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
