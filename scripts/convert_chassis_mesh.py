"""Convert Pololu's 3pi+ STEP assembly into the render meshes the plant uses.

Run this only when the meshes need regenerating; the outputs are committed, so
the project does not depend on a CAD kernel. It needs two packages that are
deliberately NOT project dependencies - install them ad hoc:

    uv pip install cadquery-ocp trimesh fast-simplification scipy

    uv run --with cadquery-ocp --with trimesh --with fast-simplification \
        python scripts/convert_chassis_mesh.py

What it does, and why each step is needed:

- Tessellates the STEP with OpenCASCADE's own visualisation mesher. gmsh was
  tried first and cannot mesh this assembly at all ("impossible to mesh periodic
  surface", then "the 1D mesh seems not to be forming a closed loop") because it
  insists on a valid conforming mesh; a render mesh does not need one.
- Splits the WHEELS out of the chassis shell. They arrive fused to the motor
  shafts as one connected component, so they are separated geometrically: a
  wheel triangle is one far out laterally and close to the axle line. They have
  to come out, or the rendered wheels would sit frozen while the robot drives.
- Rotates STEP's Y-up, +x-rearward frame into the plant's body frame (+x
  forward, +z up, origin at the wheel-axle midpoint) and scales mm -> m. The
  STEP's own origin already sits on the axle, which is worth knowing: the caster
  ball comes out at x = 40.4 mm behind it and 9.9 mm below, matching the figures
  fitted from the dimension drawing to 0.05 mm.
- Decimates. The raw tessellation is ~88k triangles, most of it screw holes and
  ribs nobody will see in a 720p overhead video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STEP = REPO_ROOT / "models/3pi-plus-chassis-kit/3pi-plus-chassis-kit-skirt.step"
DEFAULT_CHASSIS = REPO_ROOT / "models/pololu_chassis.stl"
DEFAULT_WHEEL = REPO_ROOT / "models/pololu_wheel.stl"

# STEP (mm, +y up, +x toward the caster) -> body frame (m, +x forward, +z up).
# The 3x3 block has a positive determinant, so triangle winding survives.
STEP_TO_BODY = np.array(
    [
        [-1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1e-3, 0.0],
        [0.0, 1e-3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# A triangle belongs to a wheel if it is further out than this laterally and
# closer to the axle line than this radially. Measured on the tessellation: the
# wheels occupy |lateral| 38.2-48.0 mm at a radius of at most 15.9 mm, while
# nothing else out there comes near the axle.
WHEEL_LATERAL_MM = 38.0
WHEEL_RADIUS_MM = 17.5


def tessellate(step_path: Path, deflection: float):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.StlAPI import StlAPI_Writer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"could not read {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    raw = step_path.with_suffix(".tessellated.stl")
    writer.Write(shape, str(raw))
    return raw


def main() -> int:
    import trimesh

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=Path, default=DEFAULT_STEP)
    parser.add_argument("--chassis-out", type=Path, default=DEFAULT_CHASSIS)
    parser.add_argument("--wheel-out", type=Path, default=DEFAULT_WHEEL)
    parser.add_argument("--deflection", type=float, default=0.3, help="mm of chord deviation")
    parser.add_argument("--chassis-faces", type=int, default=20000)
    parser.add_argument("--wheel-faces", type=int, default=3000)
    args = parser.parse_args()

    raw = tessellate(args.step, args.deflection)
    mesh = trimesh.load(raw)
    print(f"tessellated {mesh.faces.shape[0]} triangles")

    parts = sorted(mesh.split(only_watertight=False), key=lambda part: -part.faces.shape[0])
    # The caster ball is its own solid and gets its own geom in the XML, where it
    # is on a joint; keeping it here would freeze it inside the chassis.
    ball = min(parts, key=lambda part: abs(part.centroid[0] - 40.4) + abs(part.centroid[1] + 9.9))
    shell, others = parts[0], [part for part in parts[1:] if part is not ball]

    centre = shell.triangles_center
    radius = np.hypot(centre[:, 0], centre[:, 1])
    lateral = centre[:, 2]
    is_wheel = (np.abs(lateral) > WHEEL_LATERAL_MM) & (radius < WHEEL_RADIUS_MM)
    print(f"wheels: {int(is_wheel.sum())} triangles, chassis shell: {int((~is_wheel).sum())}")

    chassis = trimesh.util.concatenate([shell.submesh([~is_wheel], append=True), *others])
    wheel = shell.submesh([is_wheel & (lateral > 0)], append=True)
    wheel.apply_translation([0.0, 0.0, -float(np.mean(wheel.bounds[:, 2]))])

    for part, out, faces in ((chassis, args.chassis_out, args.chassis_faces), (wheel, args.wheel_out, args.wheel_faces)):
        if part.faces.shape[0] > faces:
            part = part.simplify_quadric_decimation(face_count=faces)
        part.apply_transform(STEP_TO_BODY)
        out.parent.mkdir(parents=True, exist_ok=True)
        part.export(out)
        print(f"{out.name}: {part.faces.shape[0]} triangles, bbox (m) "
              f"{np.round(part.bounds[0], 4)} .. {np.round(part.bounds[1], 4)}")
    raw.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
