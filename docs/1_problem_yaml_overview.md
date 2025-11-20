# 1. Problem YAML Overview

The simulator is driven by a YAML file such as **`empty.yaml`**.

```yaml
sim_time: 5.0
time_step: 0.01

environment:
  min: [-5, -5]
  max: [5, 5]
  obstacles: []

start: [0.0, 0.0, 0.0]
goal:  [2.0, 1.0, 1.57]

planner:
  waypoints:
    - [0.5, 0.0, -1.57]
    - [1.0, 0.5,  1.57]
  time: 4.0

controller:
  gains: [5.0, 5.0, 3.0, 0.4, 0.4, 0.2, 0.2]

robot:
  wheel_radius: 0.016
  base_diameter: 0.089
  max_wheel_speed: 40.0
  slip_r: 0.4
  slip_l: 0.5

estimator:
  type: "kf"
  wheel_radius: 0.015
  base_diameter: 0.09
  noise_pos: 0.0001
  noise_angle: 0.07
  enc_angle_noise: 0.01
  proc_pos_std: 0.7
  proc_theta_std: 0.7
  start: [0.0, 0.0, 0.0]
```

### YAML → Subsystems

| Subsystem | YAML fields | What it configures |
|----------|-------------|--------------------|
| Planner | `planner.waypoints`, `planner.time` | Path & time scaling |
| Controller | `controller.gains` | Geometric gains + PI wheel controllers |
| Robot | `robot.wheel_radius`, `robot.base_diameter`, `robot.slip_r`, `robot.slip_l`, `robot.max_wheel_speed` | True simulation dynamics |
| Estimator | `estimator.type`, noise params | DR or EKF |
| Simulation | `sim_time`, `time_step`, start/goal | Integration horizon & boundary conditions |

---

