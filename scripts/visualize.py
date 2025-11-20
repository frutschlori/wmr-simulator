import numpy as np
import yaml
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation
import matplotlib.pyplot as plt
import os 
from matplotlib.backends.backend_pdf import PdfPages

DnametoColor = {
    "red": 0xff0000,
    "green": 0x00ff00,
    "blue": 0x0000ff,
    "yellow": 0xffff00,
    "white": 0xffffff,
    "black": 0x000000,
    "cyan": 0x00ffff,
    "magenta": 0xff00ff,
    "orange": 0xffa500,
    "purple": 0x800080
}

def visualize(env_file, robot, reference_states, out_prefix="simulation_visualization"):
    vis = meshcat.Visualizer()
    anim = Animation(default_framerate=1/0.01)

    # Load environment
    with open(env_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # Extract data from robot and reference_states
    states = np.array(robot.log_states)
    
    # Extend reference states if needed to match robot states length
    if len(states) > len(reference_states):
        num_extra_steps = len(states) - len(reference_states)
        last_ref_state = reference_states[-1]
        extended_ref_states = np.vstack([
            reference_states,
            np.tile(last_ref_state, (num_extra_steps, 1))
        ])
    else:
        extended_ref_states = reference_states[:len(states)]
    
    ref_pos = extended_ref_states[:, 0:2]  # x, y
    ref_th = extended_ref_states[:, 2]     # theta
    
    obstacles = data["environment"]["obstacles"]
    for k, obs in enumerate(obstacles):
        center = obs["center"]
        size = obs["size"]
        vis[f"Obstacle_Box{k}"].set_object(g.Mesh(g.Box(size)))
        vis[f"Obstacle_Box{k}"].set_transform(tf.translation_matrix(center))

    states = np.array(states)
    px_trace, py_trace, _ = states[:,0].copy(), states[:,1].copy(), states[:,2].copy()
    px_traced, py_traced = ref_pos[:,0].copy(), ref_pos[:,1].copy()
    vis[f"trace_0"].set_object(g.Line(g.PointsGeometry( np.array([px_trace,py_trace])), g.LineBasicMaterial(color=DnametoColor.get("blue"))))
    vis[f"trace_d0"].set_object(g.Line(g.PointsGeometry( np.array([px_traced,py_traced])), g.LineBasicMaterial(color=DnametoColor.get("red"))))
    vis[f"unicycle0"].set_object(g.Mesh(g.Box([0.1, 0.05, 0.05]), material=g.MeshLambertMaterial(color=DnametoColor.get("blue"))))
    vis[f"unicycle_d0"].set_object(g.Mesh(g.Box([0.1, 0.05, 0.05]), material=g.MeshLambertMaterial(color=DnametoColor.get("red"))))
    th_ref_prev = 0.0
    for k, state in enumerate(states):
        with anim.at_frame(vis, k) as frame:
            px, py, th = state[0], state[1], state[2]
            if k < len(ref_pos):
                px_ref, py_ref = ref_pos[k,0], ref_pos[k,1]
                th_ref = ref_th[k]            

            frame[f"unicycle0"].set_transform(
                tf.translation_matrix([px, py, 0]).dot(
                    tf.quaternion_matrix(tf.quaternion_from_euler(0, 0, th))
                )
            )

            frame[f"unicycle_d0"].set_transform(
                tf.translation_matrix([px_ref, py_ref, 0]).dot(
                    tf.quaternion_matrix(tf.quaternion_from_euler(0, 0, th_ref))
                )
            )

    vis.set_animation(anim)
    res = vis.static_html()
    os.makedirs("visualize", exist_ok=True)
    video_file = os.path.join("visualize/", f"{out_prefix}.html")

    with open(video_file, "w") as f:
        f.write(res)
    print(f"Visualization saved to {video_file}")


def plot(robot, estimator, reference_states, time, out_prefix="plot_trajectory"):
    # Extract data from robot and reference_states
    states = np.array(robot.log_states)
    wheel_inputs_cmd = np.array(robot.log_wheel_cmd)
    wheel_inputs_true = np.array(robot.log_wheel_true)
    vel_omega_ctrl = np.array(robot.log_vel_omega)  # [v, w] actual from robot

    # Extract estimator logs
    est_pose_hat = np.array(estimator.log_pose_hat)   # [N, 3]
    est_pose_meas = np.array(estimator.log_pose_meas) # [N, 3]

    # Trim / extend to match states length if needed
    states = np.array(robot.log_states)
    N = len(states)
    est_pose_hat = est_pose_hat[:N]
    est_pose_meas = est_pose_meas[:N]


    # Extend reference states if needed to match time length
    if len(time) > len(reference_states):
        num_extra_steps = len(time) - len(reference_states)
        last_ref_state = reference_states[-1]
        extended_ref_states = np.vstack([
            reference_states,
            np.tile(last_ref_state, (num_extra_steps, 1))
        ])
    else:
        extended_ref_states = reference_states[:len(time)]
    
    ref_pos = extended_ref_states[:, 0:2]   # x, y
    ref_th = extended_ref_states[:, 2]      # theta
    ref_vel = extended_ref_states[:, 3:5]   # vx, vy
    ref_omega = extended_ref_states[:, 5]   # omega
    
    # Convert inputs
    ref_pos = np.array(ref_pos)
    ref_vel = np.array(ref_vel)
    ref_th = np.array(ref_th)

    refstates = np.column_stack([ref_pos, ref_th])
    # Cut reference states to match the length of time vector
    min_len = min(len(refstates), len(time))
    refstates = refstates[:min_len]
    # --- Prepare output path
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize/", f"{out_prefix}.pdf")

    with PdfPages(pdf_filename) as pdf:
        # ----------------------------------------
        # State tracking figure
        # ----------------------------------------
        fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8))
        labels = ["x", "y", "theta"]

        for k in range(3):
            axes1[k].plot(time, states[:, k], 'b-', label=f'Actual {labels[k]}')
            axes1[k].plot(time, refstates[:, k], 'r--', label=f'Ref {labels[k]}')
            axes1[k].plot(time[:N], est_pose_hat[:, k], 'g-.', label=f'Est {labels[k]}')
            axes1[k].set_ylabel(f'{["X Position", "Y Position", "Angle"][k]}')
            axes1[k].legend()
            axes1[k].grid(True)

        fig1.suptitle("State Tracking", fontsize=14)
        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig1, bbox_inches='tight', transparent=True)
        plt.close(fig1)

        # ----------------------------------------
        # Wheel inputs figure
        # ----------------------------------------
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6))

        axes2[0].plot(time, wheel_inputs_cmd[:, 0], 'g-', label='Commanded Right Wheel')
        axes2[0].plot(time, wheel_inputs_true[:, 0], 'g--', label='True Right Wheel')
        axes2[0].set_ylabel('Right Wheel Speed [rad/s]')
        axes2[0].legend()
        axes2[0].grid(True)

        axes2[1].plot(time, wheel_inputs_cmd[:, 1], 'm-', label='Commanded Left Wheel')
        axes2[1].plot(time, wheel_inputs_true[:, 1], 'm--', label='True Left Wheel')
        axes2[1].set_ylabel('Left Wheel Speed [rad/s]')
        axes2[1].set_xlabel('Time Step')
        axes2[1].legend()
        axes2[1].grid(True)

        fig2.suptitle("Wheel Speeds", fontsize=14)
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig2, bbox_inches='tight', transparent=True)
        plt.close(fig2)

        # ----------------------------------------
        # Velocity and Angular Velocity comparison
        # ----------------------------------------
        fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6))
        
        # Linear velocity comparison
        v_ctrl = vel_omega_ctrl[:, 0]
        v_reference = np.sqrt(ref_vel[:, 0]**2 + ref_vel[:, 1]**2)[:len(v_ctrl)]  # magnitude of reference velocity
        
        axes3[0].plot(time[:len(v_ctrl)], v_ctrl, 'b-', label='Controller Linear Velocity')
        axes3[0].plot(time[:len(v_reference)], v_reference, 'r--', label='Reference Linear Velocity')
        axes3[0].set_ylabel('Linear Velocity [m/s]')
        axes3[0].legend()
        axes3[0].grid(True)
        
        # Angular velocity comparison
        w_ctrl = vel_omega_ctrl[:, 1]
        w_reference = ref_omega[:len(w_ctrl)]
        
        axes3[1].plot(time[:len(w_ctrl)], w_ctrl, 'b-', label='Controller Angular Velocity')
        axes3[1].plot(time[:len(w_reference)], w_reference, 'r--', label='Reference Angular Velocity')
        axes3[1].set_ylabel('Angular Velocity [rad/s]')
        axes3[1].set_xlabel('Time [s]')
        axes3[1].legend()
        axes3[1].grid(True)
        
        fig3.suptitle("Velocity Tracking", fontsize=14)
        fig3.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig3, bbox_inches='tight', transparent=True)
        plt.close(fig3)

        # (Optional) Add metadata
        d = pdf.infodict()
        d['Title'] = 'Differential Drive Simulation Results'
        d['Author'] = 'Wheeled Robot Simulator'
        d['Subject'] = 'State Tracking and Wheel Inputs'
        d['Keywords'] = 'diffdrive, simulation, robotics, control'

    print(f"Multi-page PDF saved at: {pdf_filename}")