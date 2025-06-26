import json
import matplotlib.pyplot as plt
import numpy as np
from mpc_model_id_mismatch import helpers
from pathlib import Path

sizes = 2
widths = 0.5
n_rows_states = 4
# n_rows_states = 5
n_cols_states = 4
# plot_x_idx_at_ax_idx = [
#     0,
#     1,
#     2,
#     None,
#     3,
#     4,
#     5,
#     6,
#     7,
#     8,
#     9,
#     None,
#     10,
#     12,
#     14,
#     16,
#     11,
#     13,
#     15,
#     None,
# ]
# plot_y_idx_at_ax_idx = [
#     0,
#     1,
#     2,
#     None,
#     3,
#     4,
#     5,
#     6,
#     7,
#     8,
#     9,
#     None,
#     10,
#     11,
#     12,
#     13,
#     None,
#     None,
#     None,
#     None,
# ]
# plot_u_idx_at_ax_idx = [
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     0,
#     1,
#     2,
#     3,
#     None,
#     None,
#     None,
#     None,
# ]
# plot_w_idx_at_ax_idx = [
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     0,
#     1,
#     2,
#     None,
#     3,
#     5,
#     7,
#     9,
#     4,
#     6,
#     8,
#     None,
# ]
# x_labels = [
#     "px (m)",
#     "py (m)",
#     "pz (m)",
#     "qw (-)",
#     "qx (-)",
#     "qy (-)",
#     "qz (-)",
#     "vx (m/s)",
#     "vy (m/s)",
#     "vz (m/s)",
#     "wbx (rad/s)",
#     "wbx1 (rad/s^2?)",
#     "wby (rad/s)",
#     "wby1 (rad/s^2?)",
#     "wbz (rad/s)",
#     "wbz1 (rad/s^2?)",
#     "abz (m/s^2)",
# ]
# y_labels = [
#     "px (m)",
#     "py (m)",
#     "pz (m)",
#     "qw (-)",
#     "qx (-)",
#     "qy (-)",
#     "qz (-)",
#     "vx (m/s)",
#     "vy (m/s)",
#     "vz (m/s)",
#     "wbx (rad/s)",
#     "wby (rad/s)",
#     "wbz (rad/s)",
#     "abz (m/s^2)",
# ]
# w_labels = [
#     "$w_{vx} (m/s / s)$",
#     "$w_{vy} (m/s / s)$",
#     "$w_{vz} (m/s / s)$",
#     "$w_{wbx} (rad/s / s)$",
#     "$w_{wbx1} (rad/s^2? / s)$",
#     "$w_{wby} (rad/s / s)$",
#     "$w_{wby1} (rad/s^2? / s)$",
#     "$w_{wbz} (rad/s / s)$",
#     "$w_{wbz1} (rad/s^2? / s)$",
#     "$w_{abz} (m/s^2 / s)$",
# ]
plot_x_idx_at_ax_idx = [
    0,
    1,
    2,
    None,
    3,
    4,
    5,
    None,
    6,
    7,
    8,
    None,
    9,
    10,
    11,
    None,
]
plot_y_idx_at_ax_idx = plot_x_idx_at_ax_idx
plot_u_idx_at_ax_idx = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    0,
    1,
    2,
    3,
]
plot_w_idx_at_ax_idx = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    0,
    1,
    2,
    None,
    3,
    4,
    5,
    None,
]
plot_eta_idx_at_ax_idx = [
    0,
    1,
    2,
    None,
    3,
    4,
    5,
    None,
    6,
    7,
    8,
    None,
    9,
    10,
    11,
    None,
]
x_labels = [
    "px (m)",
    "py (m)",
    "pz (m)",
    "$\phi$ (rad)",
    "$\\theta$ (rad)",
    "$\psi$ (rad)",
    "vx (m/s)",
    "vy (m/s)",
    "vz (m/s)",
    "wbx (rad/s)",
    "wby (rad/s)",
    "wbz (rad/s)",
]
y_labels = x_labels
w_labels = [
    "$w_{vx} (m/s / s)$",
    "$w_{vy} (m/s / s)$",
    "$w_{vz} (m/s / s)$",
    "$w_{wbx} (rad/s / s)$",
    "$w_{wby} (rad/s / s)$",
    "$w_{wbz} (rad/s / s)$",
]
# plot_x_idx_at_ax_idx = [
#     0,
#     1,
#     2,
#     None,
#     3,
#     4,
#     5,
#     None,
#     6,
#     7,
#     8,
#     None,
#     9,
#     10,
#     11,
#     None,
#     12,
#     13,
#     14,
#     15,
# ]
# plot_y_idx_at_ax_idx = plot_x_idx_at_ax_idx
# plot_u_idx_at_ax_idx = [
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     0,
#     1,
#     2,
#     3,
# ]
# plot_w_idx_at_ax_idx = [
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     0,
#     1,
#     2,
#     None,
#     3,
#     4,
#     5,
#     None,
#     6,
#     7,
#     8,
#     9,
# ]
# plot_eta_idx_at_ax_idx = [
#     0,
#     1,
#     2,
#     None,
#     3,
#     4,
#     5,
#     None,
#     6,
#     7,
#     8,
#     None,
#     9,
#     10,
#     11,
#     None,
#     None,
#     None,
#     None,
#     None,
# ]
# x_labels = [
#     "px (m)",
#     "py (m)",
#     "pz (m)",
#     "$\phi$ (rad)",
#     "$\\theta$ (rad)",
#     "$\psi$ (rad)",
#     "vx (m/s)",
#     "vy (m/s)",
#     "vz (m/s)",
#     "wbx (rad/s)",
#     "wby (rad/s)",
#     "wbz (rad/s)",
#     "wm0 (rad/s)",
#     "wm1 (rad/s)",
#     "wm2 (rad/s)",
#     "wm3 (rad/s)",
# ]
# y_labels = x_labels
# w_labels = [
#     "$w_{vx} (m/s / s)$",
#     "$w_{vy} (m/s / s)$",
#     "$w_{vz} (m/s / s)$",
#     "$w_{wbx} (rad/s / s)$",
#     "$w_{wby} (rad/s / s)$",
#     "$w_{wbz} (rad/s / s)$",
#     "$w_{wm0} (rad/s / s)$",
#     "$w_{wm1} (rad/s / s)$",
#     "$w_{wm2} (rad/s / s)$",
#     "$w_{wm3} (rad/s / s)$",
# ]


def get_x_y_fs(model, ts, M, u, x_est, w_est, eta_est, time_idx):
    n_horizon = M + 1
    n_x = x_est.shape[1]
    n_y = n_x
    x_fs = np.zeros((n_horizon, n_x))
    y_fs = np.zeros((n_horizon, n_y))
    x_fs[0, :] = x_est[time_idx, :, 0]
    y_fs[0, :] = model.get_outputs_noise(
        x_fs[0, :], u[:, time_idx], eta_est[time_idx, :, 0]
    )
    for k in range(M):
        x_fs[k + 1, :] = np.array(
            helpers.solve_rk4_noise(
                model.state_update_ct_noise,
                x_est[time_idx, :, k],
                u[:, time_idx + k],
                w_est[time_idx, :, k],
                ts,
            )
        ).reshape((-1,))
        # x_fs[k + 1, :] = np.array(
        #     helpers.solve_rk4(
        #         model.state_update_ct,
        #         x_fs[k, :],
        #         u[:, time_idx + k],
        #         ts,
        #     )
        # ).reshape((-1,))
        # y_fs[k + 1, :] = model.get_outputs(x_fs[k + 1, :], u[:, time_idx + k + 1])
        y_fs[k + 1, :] = model.get_outputs_noise(
            x_fs[k + 1, :], u[:, time_idx + k + 1], eta_est[time_idx, :, k + 1]
        )

    return x_fs, y_fs


def get_thrusts_torques_over_horizon(model, u, x_est, M, time_idx):
    n_horizon = M + 1

    # Compute estimated thrust and torques based on individual rotor thrusts
    thrusts_horizon = np.zeros((n_horizon, 1))
    torques_horizon = np.zeros((n_horizon, 3))
    for k_idx in range(n_horizon):
        thrusts_horizon[k_idx, :] = model.state_update_ct_compute_thrust_t(
            x_est[time_idx, :, k_idx], u[:, time_idx + k_idx]
        )
        torques_horizon[k_idx, :] = np.array(
            model.state_update_ct_compute_torque_t(
                x_est[time_idx, :, k_idx], u[:, time_idx + k_idx]
            )
        ).reshape((-1,))
    return thrusts_horizon, torques_horizon


def plot_y_y_fs_over_horizon(exp_idx, t, y, y_fs, M, time_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Measured outputs vs forward-simulated outputs over horizon",
    )
    fig.suptitle(
        f"Measured outputs vs forward-simulated outputs over horizon at time t={t[time_idx]}s (time_idx={time_idx}/{t.shape[0]})"
    )
    n_horizon = M + 1
    t = (np.arange(0, n_horizon) * ts).reshape((n_horizon,))
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_x_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        if plot_y_idx_at_ax_idx[ax_idx] != None:
            y_idx = plot_y_idx_at_ax_idx[ax_idx]
            axes[row_idx, col_idx].plot(
                t,
                y[y_idx, time_idx : time_idx + n_horizon],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
            axes[row_idx, col_idx].plot(
                t,
                y_fs[:, y_idx],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel(y_labels[y_idx])
    fig.legend(["Measured", "Forward-simulated"])


def plot_y_x_est_u_over_time(exp_idx, t, x_est, y, u, stage_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Measured outputs vs estimated states over time",
    )
    fig.suptitle(
        f"Measured outputs vs estimated states over time at stage k={stage_idx}"
    )
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_x_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        x_idx = plot_x_idx_at_ax_idx[ax_idx]
        if plot_y_idx_at_ax_idx[ax_idx] != None:
            y_idx = plot_y_idx_at_ax_idx[ax_idx]
            axes[row_idx, col_idx].plot(
                t,
                y[y_idx, stage_idx : stage_idx + len(t)],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].plot(
            t,
            x_est[:, x_idx, stage_idx],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        if plot_u_idx_at_ax_idx[ax_idx] != None:
            u_idx = plot_u_idx_at_ax_idx[ax_idx]
            axes[row_idx, col_idx].plot(
                t,
                u[u_idx, stage_idx : stage_idx + len(t)],
                "-+",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel(x_labels[x_idx])
    fig.legend(["Measured outputs", "Estimated states", "Applied inputs"])


def plot_y_x_est_u_over_horizon(exp_idx, t, x_est_all, y, u, time_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Measured outputs vs estimated states over horizon",
    )
    fig.suptitle(
        f"Measured outputs vs estimated states over horizon at time t={t[time_idx]}s (time_idx={time_idx}/{t.shape[0]})"
    )
    n_horizon = x_est_all.shape[2]
    t = (np.arange(0, n_horizon) * ts).reshape((n_horizon,))
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_x_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        x_idx = plot_x_idx_at_ax_idx[ax_idx]
        if plot_y_idx_at_ax_idx[ax_idx] != None:
            y_idx = plot_y_idx_at_ax_idx[ax_idx]
            axes[row_idx, col_idx].plot(
                t,
                y[y_idx, time_idx : time_idx + n_horizon],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].plot(
            t,
            x_est_all[time_idx, x_idx, :],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        if plot_u_idx_at_ax_idx[ax_idx] != None:
            u_idx = plot_u_idx_at_ax_idx[ax_idx]
            axes[row_idx, col_idx].plot(
                t,
                u[u_idx, time_idx : time_idx + n_horizon],
                "-+",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel(x_labels[x_idx])
    fig.legend(["Measured outputs", "Estimated states", "Applied inputs"])


def plot_u_t_over_horizon(exp_idx, t, thrusts_horizon, torques_horizon, M, time_idx):
    n_horizon = M + 1

    # Plot the motor velocity inputs, states, torques and thrust
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Applied inputs over horizon",
    )
    fig.suptitle(
        f"Applied inputs over horizon at time t={t[time_idx]}s (time_idx={time_idx}/{t.shape[0]})"
    )
    t = (np.arange(0, n_horizon) * ts).reshape((n_horizon,))
    for ax_idx in range(n_rows_states * n_cols_states):
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        if ax_idx in [8, 9, 10]:
            axes[row_idx, col_idx].plot(
                t,
                thrusts_horizon,
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        elif ax_idx in [12, 13, 14]:
            axes[row_idx, col_idx].plot(
                t,
                torques_horizon[:, ax_idx - 12],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel("Equiv. inputs")


def plot_u_wm_over_horizon(exp_idx, model, t, x_est, u, M, time_idx):
    n_horizon = M + 1

    # Compute estimated thrust and torques based on motor velocities
    thrust_pred = np.zeros((n_horizon, 1))
    torques_pred = np.zeros((n_horizon, 3))
    for k_idx in range(n_horizon):
        thrust_pred[k_idx, :] = model.state_update_ct_compute_thrust_wm(
            x_est[time_idx, :, k_idx], u[:, time_idx + k_idx]
        )
        torques_pred[k_idx, :] = np.array(
            model.state_update_ct_compute_torque_wm(
                x_est[time_idx, :, k_idx], u[:, time_idx + k_idx]
            )
        ).reshape((-1,))

    # Plot the motor velocity inputs, states, torques and thrust
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Applied inputs over horizon",
    )
    fig.suptitle(
        f"Applied inputs over horizon at time t={t[time_idx]}s (time_idx={time_idx}/{t.shape[0]})"
    )
    t = (np.arange(0, n_horizon) * ts).reshape((n_horizon,))
    for ax_idx in range(n_rows_states * n_cols_states):
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        if ax_idx in [8, 9, 10]:
            axes[row_idx, col_idx].plot(
                t,
                thrust_pred,
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        elif ax_idx in [12, 13, 14]:
            axes[row_idx, col_idx].plot(
                t,
                torques_pred[:, ax_idx - 12],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        elif ax_idx in [16, 17, 18, 19]:
            axes[row_idx, col_idx].plot(
                t,
                u[ax_idx - 16, time_idx : time_idx + n_horizon],
                "-+",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel("Equiv. inputs")


def plot_w_est_over_time(exp_idx, t, w, w_est, stage_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Estimated disturbances over time",
    )
    fig.suptitle(f"Estimated disturbances over time at stage k={stage_idx}")
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_w_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        w_idx = plot_w_idx_at_ax_idx[ax_idx]
        if w is not None:
            axes[row_idx, col_idx].plot(
                t,
                w[w_idx, stage_idx : stage_idx + len(t)],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].plot(
            t,
            w_est[:, w_idx, stage_idx],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel(w_labels[w_idx])
    if w is not None:
        fig.legend(["Ground truth", "Estimated"])


def plot_w_est_over_horizon(exp_idx, t, ts, w, w_est, time_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Estimated disturbances over horizon",
    )
    fig.suptitle(
        f"Estimated disturbances over horizon at time t={t[time_idx]}s (time_idx={time_idx}/{t.shape[0]})"
    )
    M = w_est.shape[2]
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_w_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        w_idx = plot_w_idx_at_ax_idx[ax_idx]
        if w is not None:
            axes[row_idx, col_idx].plot(
                np.squeeze(np.arange(0, M) * ts),
                w[w_idx, time_idx : time_idx + M],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].plot(
            np.squeeze(np.arange(0, M) * ts),
            w_est[time_idx, w_idx, :],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel(w_labels[w_idx])
    if w is not None:
        fig.legend(["Ground truth", "Estimated"])


def plot_eta_est_over_time(exp_idx, t, eta, eta_est, stage_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Estimated measurement noises over time",
    )
    fig.suptitle(f"Estimated measurement noises over time at stage k={stage_idx}")
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_eta_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        eta_idx = plot_eta_idx_at_ax_idx[ax_idx]
        if eta is not None:
            axes[row_idx, col_idx].plot(
                t,
                eta[eta_idx, stage_idx : stage_idx + len(t)],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].plot(
            t,
            eta_est[:, eta_idx, stage_idx],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel(y_labels[eta_idx])
    if eta is not None:
        fig.legend(["Ground truth", "Estimated"])


def plot_eta_est_over_horizon(exp_idx, t, ts, eta, eta_est, time_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Estimated measurement noises over horizon",
    )
    fig.suptitle(
        f"Estimated measurement noises over horizon at time t={t[time_idx]}s (time_idx={time_idx}/{t.shape[0]})"
    )
    n_horizon = eta_est.shape[2]
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_eta_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        eta_idx = plot_eta_idx_at_ax_idx[ax_idx]
        if eta is not None:
            axes[row_idx, col_idx].plot(
                np.squeeze(np.arange(0, n_horizon) * ts),
                eta[eta_idx, time_idx : time_idx + n_horizon],
                "-o",
                linewidth=widths,
                markersize=sizes,
            )
        axes[row_idx, col_idx].plot(
            np.squeeze(np.arange(0, n_horizon) * ts),
            eta_est[time_idx, eta_idx, :],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Time (s)")
        axes[row_idx, col_idx].set_ylabel(y_labels[eta_idx])
    if eta is not None:
        fig.legend(["Ground truth", "Estimated"])


def plot_x_est_w_est_stage(exp_idx, x_est, w_est, stage_idx):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Estimated states vs estimated disturbances over time",
    )
    fig.suptitle(
        f"Estimated states vs estimated disturbances over time at stage k={stage_idx}"
    )
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_w_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        w_idx = plot_w_idx_at_ax_idx[ax_idx]
        axes[row_idx, col_idx].scatter(
            x_est[:, 6 + w_idx, stage_idx],
            w_est[:, w_idx, stage_idx],
            s=sizes,
            linewidth=widths,
        )
        axes[row_idx, col_idx].set_xlabel(f"{x_labels[6 + w_idx]}")
        axes[row_idx, col_idx].set_ylabel(f"{w_labels[w_idx]}")


def plot_Q_R_trace(exp_idx, Q_cov_est_all, R_cov_est_all):
    fig, axes = plt.subplots(
        1,
        2,
        num=f"Experiment {exp_idx} - Trace of Q and R matrices over iterations",
    )
    fig.suptitle(f"Trace of Q and R matrices over iterations")
    fig.subplots_adjust(wspace=0.4)
    axes[0].plot(
        np.arange(0, Q_cov_est_all.shape[0]),
        np.trace(Q_cov_est_all, axis1=1, axis2=2),
        "-o",
        linewidth=widths,
        markersize=sizes,
    )
    axes[0].set_title("trace(Q)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")
    axes[1].plot(
        np.arange(0, R_cov_est_all.shape[0]),
        np.trace(R_cov_est_all, axis1=1, axis2=2),
        "-o",
        linewidth=widths,
        markersize=sizes,
    )
    axes[1].set_title("trace(R)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")


def plot_Q_diag(exp_idx, Q_cov_est_all):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Diagonal of Q matrix over iterations",
    )
    fig.suptitle(f"Diagonal of Q matrix over iterations")
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_w_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        w_idx = plot_w_idx_at_ax_idx[ax_idx]
        axes[row_idx, col_idx].plot(
            np.arange(0, Q_cov_est_all.shape[0]),
            Q_cov_est_all[:, w_idx, w_idx],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Iteration")
        axes[row_idx, col_idx].set_ylabel(f"Q[{w_idx}, {w_idx}]")


def plot_R_diag(exp_idx, R_cov_est_all):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Diagonal of R matrix over iterations",
    )
    fig.suptitle(f"Diagonal of R matrix over iterations")
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_eta_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        eta_idx = plot_eta_idx_at_ax_idx[ax_idx]
        axes[row_idx, col_idx].plot(
            np.arange(0, R_cov_est_all.shape[0]),
            R_cov_est_all[:, eta_idx, eta_idx],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Iteration")
        axes[row_idx, col_idx].set_ylabel(f"R[{eta_idx}, {eta_idx}]")


def plot_Q_eig_vals(exp_idx, Q_cov_est_all):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Eigenvalues of Q matrix over iterations",
    )
    fig.suptitle(f"Eigenvalues of Q matrix over iterations")
    eig_vals = np.zeros((Q_cov_est_all.shape[1], Q_cov_est_all.shape[0]))
    for i in range(Q_cov_est_all.shape[0]):
        eig_vals[:, i] = np.linalg.eigvals(Q_cov_est_all[i, :, :])
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_w_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        w_idx = plot_w_idx_at_ax_idx[ax_idx]
        axes[row_idx, col_idx].plot(
            np.arange(0, eig_vals.shape[1]),
            eig_vals[w_idx, :],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Iteration")
        axes[row_idx, col_idx].set_ylabel(f"Eigenvalue {w_idx}")


def plot_R_eig_vals(exp_idx, R_cov_est_all):
    fig, axes = plt.subplots(
        n_rows_states,
        n_cols_states,
        num=f"Experiment {exp_idx} - Eigenvalues of R matrix over iterations",
    )
    fig.suptitle(f"Eigenvalues of R matrix over iterations")
    eig_vals = np.zeros((R_cov_est_all.shape[1], R_cov_est_all.shape[0]))
    for i in range(R_cov_est_all.shape[0]):
        eig_vals[:, i] = np.linalg.eigvals(R_cov_est_all[i, :, :])
    for ax_idx in range(n_rows_states * n_cols_states):
        if plot_eta_idx_at_ax_idx[ax_idx] == None:
            axes.flat[ax_idx].axis("off")
            continue
        row_idx = ax_idx // n_cols_states
        col_idx = ax_idx % n_cols_states
        eta_idx = plot_eta_idx_at_ax_idx[ax_idx]
        axes[row_idx, col_idx].plot(
            np.arange(0, eig_vals.shape[1]),
            eig_vals[eta_idx, :],
            "-o",
            linewidth=widths,
            markersize=sizes,
        )
        axes[row_idx, col_idx].set_xlabel("Iteration")
        axes[row_idx, col_idx].set_ylabel(f"Eigenvalue {eta_idx}")


if __name__ == "__main__":
    # Ensure that matrices are fully printed
    np.set_printoptions(threshold=np.inf)

    # Load data from falcon_t.json
    package_dir = Path(__file__).parents[1]
    model_mismatch_results_dir = f"{package_dir}/data/model_mismatch_results"
    with open(f"{model_mismatch_results_dir}/falcon_t.json", "r") as f:
        data = json.load(f)

    # Get common data
    data_common = data["common"]
    g = data_common["g"]
    ts = data_common["ts"]
    params_file = data_common["params_file"]
    model_name = data_common["model_name"]
    M = data_common["M"]
    stage_est = data_common["stage_est"]
    nx = data_common["nx"]
    nw = data_common["nw"]
    neta = data_common["neta"]

    # Select experiments
    exp_idc = [0]
    exp_names = []
    for key in data.keys():
        if key not in ["__header__", "__version__", "__globals__", "common"]:
            exp_names.append(key)
    if len(exp_idc) > len(exp_names):
        raise ValueError(
            f"Number of selected experiments ({len(exp_idc)}) exceeds number of available experiments ({len(exp_names)})"
        )
    for exp_idx in exp_idc:
        exp_name = exp_names[exp_idx]
        print(f"Experiment {exp_idx}: {exp_name}")

        # Get experiment data
        data_exp = data[exp_name]
        t = np.array(data_exp["t"]).reshape((-1,))
        y = np.array(data_exp["y"])
        u = np.array(data_exp["u"])
        w = None
        if "w" in data_exp:
            w = np.array(data_exp["w"])
        eta = None
        if "eta" in data_exp:
            eta = np.array(data_exp["eta"])
        x_est_all = np.array(data_exp["x_est_all"])
        w_est_all = np.array(data_exp["w_est_all"])
        eta_est_all = np.array(data_exp["eta_est_all"])
        n_iter = x_est_all.shape[0]
        n_t = x_est_all.shape[1]
        t = t[stage_est : stage_est + n_t]

        # Create model class
        model = helpers.DroneAgiModel(model_name, g, params_file)

        # Set data indices to use
        iter_idx = n_iter - 1
        stage_idx = stage_est
        time_idx = 0

        # Print estimated Q and R matrices
        Q_cov_est_all = np.array(data_exp["Q_cov_est_all"])
        R_cov_est_all = np.array(data_exp["R_cov_est_all"])
        # print(f"Q_est = {Q_cov_est_all[iter_idx + 1, :, :]}")
        # print(f"R_est = {R_cov_est_all[iter_idx + 1, :, :]}")

        # Select data for the selected iteration
        x_est_all = x_est_all[iter_idx, :, :, :]
        w_est_all = w_est_all[iter_idx, :, :, :]
        eta_est_all = eta_est_all[iter_idx, :, :, :]

        # Print estimated disturbances and measurement noise
        # print(f"w_est = {w_est_all[0, :, :]}")
        # print(f"eta_est = {eta_est_all[0, :, :]}")

        # Compute forward-simulated outputs and thrusts/torques over horizon
        x_fs, y_fs = get_x_y_fs(
            model, ts, M, u, x_est_all, w_est_all, eta_est_all, time_idx
        )
        thrusts_horizon, torques_horizon = get_thrusts_torques_over_horizon(
            model, u, x_est_all, M, time_idx
        )

        # Create plots
        # plot_y_y_fs_over_horizon(exp_idx, t, y, y_fs, M, time_idx)
        # plot_y_x_est_u_over_time(exp_idx, t, x_est_all, y, u, stage_idx)
        # plot_y_x_est_u_over_horizon(exp_idx, t, x_est_all, y, u, time_idx)
        # plot_u_t_over_horizon(exp_idx, t, thrusts_horizon, torques_horizon, M, time_idx)
        # plot_u_wm_over_horizon(exp_idx, model, t, x_est_all, u, M, time_idx)
        # plot_w_est_over_time(exp_idx, t, w, w_est_all, stage_idx)
        plot_w_est_over_horizon(exp_idx, t, ts, w, w_est_all, time_idx)
        # plot_eta_est_over_time(exp_idx, t, eta, eta_est_all, stage_idx)
        plot_eta_est_over_horizon(exp_idx, t, ts, eta, eta_est_all, time_idx)
        # plot_x_est_w_est_stage(exp_idx, x_est_all, w_est_all, stage_idx)
        # plot_Q_R_trace(exp_idx, Q_cov_est_all, R_cov_est_all)
        # plot_Q_diag(exp_idx, Q_cov_est_all)
        # plot_R_diag(exp_idx, R_cov_est_all)
        plot_Q_eig_vals(exp_idx, Q_cov_est_all)
        plot_R_eig_vals(exp_idx, R_cov_est_all)
    plt.show()
