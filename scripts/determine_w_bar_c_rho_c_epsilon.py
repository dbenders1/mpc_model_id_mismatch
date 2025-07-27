import argparse
import json
import logging
import math
import yaml

import matplotlib.pyplot as plt
import numpy as np

from os import path
from pathlib import Path
from mpc_model_id_mismatch import helpers

if __name__ == "__main__":
    # Log settings
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="something")
    parser.add_argument("-v", "--verbose", action="count", default=0, dest="verbosity")
    args = parser.parse_args()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN - 10 * args.verbosity)

    # User settings
    package_dir = Path(__file__).parents[1]
    runtime_json_dir = f"{package_dir}/../rmpc/mpc_tools/recorded_data"
    if not path.exists(runtime_json_dir):
        log.warning(
            f"Directory {runtime_json_dir} does not exist! Please ensure that the rmpc submodule is cloned"
        )
        exit(1)
    ros_rec_json_dir = f"{package_dir}/../rosbag2json/data/converted_bags"
    if not path.exists(ros_rec_json_dir):
        log.warning(
            f"Directory {ros_rec_json_dir} does not exist! Please ensure that the rosbag2json submodule is cloned"
        )
        exit(1)
    config_dir = f"{package_dir}/config"
    config_path = f"{config_dir}/scripts/determine_w_bar_c_rho_c_epsilon.yaml"
    data_dir = f"{package_dir}/data"
    data_sel_dir = f"{data_dir}/selected_data"

    # Read configuration parameters
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    g = config["constants"]["g"]

    runtime_json_names = config["data"]["runtime_json_names"]
    ros_rec_json_names = config["data"]["ros_rec_json_names"]
    use_nominal_reference = config["data"]["use_nominal_reference"]
    n_idx_ignore = config["data"]["n_idx_ignore"]

    compute_settings = config["compute_settings"]
    compute_rho_c = compute_settings["rho_c"]

    do_plot = config["do_plot"]
    do_plot_lyap_err = do_plot["lyap_err"]
    do_plot_rpi_tightening_over_rho_c = do_plot["rpi_tightening_over_rho_c"]
    do_plot_w_bar_c_time = do_plot["w_bar_c_time"]
    do_plot_w_bar_c_sorted = do_plot["w_bar_c_sorted"]
    do_plot_epsilon_time = do_plot["epsilon_time"]
    do_plot_epsilon_sorted = do_plot["epsilon_sorted"]

    quad_name = config["model"]["name"]

    plot_settings = config["plot_settings"]
    linewidth = plot_settings["linewidth"]
    n_rows_plot = plot_settings["n_rows"]
    n_cols_plot = plot_settings["n_cols"]
    plot_stage_idx_at_ax_idx = plot_settings["plot_stage_idx_at_ax_idx"]

    # Print warning if the number of runtime and ros recording json names do not match
    n_runtime_json_names = len(runtime_json_names)
    n_ros_rec_json_names = len(ros_rec_json_names)
    if n_runtime_json_names != n_ros_rec_json_names:
        raise ValueError(
            f"Number of runtime_json_names ({n_runtime_json_names}) and ros_rec_json_names ({n_ros_rec_json_names}) must match"
        )

    # Create model
    quad_name = config["model"]["name"]
    g = config["constants"]["g"]
    params_file = f"{config_dir}/systems/{quad_name}.yaml"
    if path.exists(params_file):
        log.warning(f"Selected {quad_name} params file")
    else:
        log.fatal(f"Unknown quad name {quad_name}! Exiting")
        exit(1)
    if quad_name == "falcon":
        model = helpers.DroneAgiModel(quad_name, g, params_file)
    else:
        log.fatal(f"Unknown model {quad_name}! Exiting")
        exit(1)

    # Iterate over all runtime and ros recording json names
    for file_idx in range(n_runtime_json_names):
        runtime_json_name = runtime_json_names[file_idx]
        ros_rec_json_name = ros_rec_json_names[file_idx]

        # Read ROS runtime json data
        runtime_json_path = f"{runtime_json_dir}/{runtime_json_name}.json"
        log.warning(f"Selected runtime json file: {runtime_json_path}")
        if not path.exists(runtime_json_path):
            raise FileNotFoundError(f"Runtime json path {runtime_json_path} not found")
        with open(runtime_json_path, "r") as file:
            runtime_data = json.load(file)

        runtime_static_data = runtime_data["static_data"]
        stepsize_tmpc = runtime_static_data["stepsize"]
        steps_tmpc = runtime_static_data["steps"]
        P_delta = np.array(runtime_static_data["P_delta"])
        rho_c = np.array(runtime_static_data["rho_c"])

        # Read ROS recording json data
        ros_rec_json_path = f"{ros_rec_json_dir}/{ros_rec_json_name}.json"
        log.warning(f"Selected ROS recording json file: {ros_rec_json_path}")
        if not path.exists(ros_rec_json_path):
            raise FileNotFoundError(
                f"ROS recording json path {ros_rec_json_path} not found"
            )
        with open(ros_rec_json_path, "r") as file:
            ros_rec_data = json.load(file)

        time_precision = ros_rec_data["time_precision"]

        data_x_cur = ros_rec_data["/falcon/ground_truth/odometry"]
        t_x_cur = np.array(data_x_cur["t"])
        x_cur = np.array(data_x_cur["x"])

        data_x_cur_est = ros_rec_data["/mpc/rec/current_state"]
        t_x_cur_est = np.array(data_x_cur_est["t"])
        x_cur_est = np.array(data_x_cur_est["current_state"])

        data_pred_traj = ros_rec_data["/mpc/rec/predicted_trajectory/0"]
        t_pred_traj = np.array(data_pred_traj["t"])
        u_pred_traj = np.array(data_pred_traj["u_pred"])
        x_pred_traj = np.array(data_pred_traj["x_pred"])

        if use_nominal_reference:
            data_nom_ref = ros_rec_data["/mpc/rec/nominal_reference"]
            t_nom_ref = np.array(data_nom_ref["t"])
            x_nom_ref = np.array(data_nom_ref["x_ref"])
            u_nom_ref = np.array(data_nom_ref["u_ref"])

        # Set times to a specific precision
        t_x_cur_est = np.round(t_x_cur_est, time_precision)
        t_pred_traj = np.round(t_pred_traj, time_precision)
        if use_nominal_reference:
            t_nom_ref = np.round(t_nom_ref, time_precision)

        # Determine various parameters of runtime data
        nx = x_cur.shape[1]
        n_tmpc = min(len(t_x_cur_est), len(t_pred_traj))
        if use_nominal_reference:
            n_tmpc = min(len(t_x_cur_est), len(t_pred_traj), len(t_nom_ref))
        N_tmpc = x_pred_traj.shape[1] - 1
        dt_tmpc = steps_tmpc * stepsize_tmpc

        # Align all data recorded in mpc
        t_x_cur_est = t_x_cur_est[:n_tmpc]
        x_cur_est = x_cur_est[:n_tmpc, :]
        x_cur_est = x_cur_est[:n_tmpc, :]
        t_pred_traj = t_pred_traj[:n_tmpc]
        u_pred_traj = u_pred_traj[:n_tmpc, :, :]
        x_pred_traj = x_pred_traj[:n_tmpc, :, :]
        if use_nominal_reference:
            t_nom_ref = t_nom_ref[:n_tmpc]
            x_nom_ref = x_nom_ref[:n_tmpc]
            u_nom_ref = u_nom_ref[:n_tmpc]

        # Ensure that all times are aligned
        if t_x_cur_est[-1] > t_x_cur[-1]:
            log.warning(
                "The estimated state has a recording after the ground truth state. Shrinking t_x_cur_est, x_cur_est, and (t_pred_traj, u_pred_traj, x_pred_traj) or (t_nom_ref, u_nom_ref, x_nom_ref) to the last time of t_x_cur"
            )
            t_x_cur_est = t_x_cur_est[t_x_cur_est <= t_x_cur[-1]]
            x_cur_est = x_cur_est[: len(t_x_cur_est)]
            t_pred_traj = t_pred_traj[t_pred_traj <= t_x_cur[-1]]
            u_pred_traj = u_pred_traj[: len(t_pred_traj)]
            x_pred_traj = x_pred_traj[: len(t_pred_traj)]
            if use_nominal_reference:
                t_nom_ref = t_nom_ref[t_nom_ref <= t_x_cur[-1]]
                u_nom_ref = u_nom_ref[: len(t_nom_ref)]
                x_nom_ref = x_nom_ref[: len(t_nom_ref)]
        print(f"t start: {t_x_cur_est[0]}")
        print(f"t end: {t_x_cur_est[-1]}")

        # When computing rho_c, we want to compute w_bar_c over a uniform grid of rho_c values
        if not compute_rho_c:
            rho_c_all = np.array([rho_c])
        else:
            # Forward-simulate the system for n_forward_sim steps
            t_forward_sim = 1
            n_forward_sim = int(t_forward_sim / dt_tmpc)
            n_times = n_tmpc - n_idx_ignore - n_forward_sim
            x_forward_sim = np.zeros((n_times, 1 + n_forward_sim, nx))
            for t_idx in range(n_times):
                if t_idx < n_times - 1:
                    print(
                        f"Forward simulating time iter {t_idx}/{n_times - 1}", end="\r"
                    )
                else:
                    print(f"Forward simulating time iter {t_idx}/{n_times - 1}")
                if not use_nominal_reference:
                    x_forward_sim[t_idx, 0] = x_cur_est[n_idx_ignore + t_idx]
                    for k_idx in range(n_forward_sim):
                        x_forward_sim[t_idx, k_idx + 1] = np.array(
                            helpers.solve_rk4(
                                model.state_update_ct,
                                x_forward_sim[t_idx, k_idx],
                                u_pred_traj[n_idx_ignore + t_idx, 0],
                                dt_tmpc,
                            )
                        ).flatten()
                else:
                    x_forward_sim[t_idx, 0] = x_nom_ref[n_idx_ignore + t_idx]
                    for k_idx in range(n_forward_sim):
                        x_forward_sim[t_idx, k_idx + 1] = np.array(
                            helpers.solve_rk4(
                                model.state_update_ct,
                                x_forward_sim[t_idx, k_idx],
                                u_nom_ref[n_idx_ignore + t_idx],
                                dt_tmpc,
                            )
                        ).flatten()
            data_x_fs = {"x_forward_sim": x_forward_sim.tolist()}
            with open("x_forward_sim.json", "w") as f:
                json.dump(
                    data_x_fs,
                    f,
                )
            # with open("x_forward_sim.json", "r") as openfile:
            #     x_forward_sim_dict = json.load(openfile)
            #     x_forward_sim = np.array(x_forward_sim_dict["x_forward_sim"])
            # n_times = x_forward_sim.shape[0]
            # n_forward_sim = x_forward_sim.shape[1] - 1

            # Compute x_err and lyap_err over times and prediction steps
            x_err = np.zeros((n_times, n_forward_sim, nx))
            lyap_err = np.zeros((n_times, n_forward_sim))
            for t_idx in range(n_times):
                if t_idx < n_times - 1:
                    print(f"Time iter {t_idx}/{n_times - 1}", end="\r")
                else:
                    print(f"Time iter {t_idx}/{n_times - 1}")
                for k_idx in range(n_forward_sim):
                    # x_err[t_idx, k_idx] = (
                    #     x_cur_est[n_idx_ignore + t_idx + 1 + k_idx]
                    #     - x_pred_traj[n_idx_ignore + t_idx, 1 + k_idx]
                    # )
                    x_err[t_idx, k_idx] = (
                        x_cur_est[n_idx_ignore + t_idx + 1 + k_idx]
                        - x_forward_sim[t_idx, 1 + k_idx]
                    )
                    lyap_err[t_idx, k_idx] = np.sqrt(
                        x_err[t_idx, k_idx] @ P_delta @ x_err[t_idx, k_idx]
                    )

            # Compute w_bar_c for all rho_c, t, and tau values
            n_rho_c_all = 1000
            rho_c_all = np.linspace(0.01, 100, n_rho_c_all)
            w_bar_c_all = np.zeros((n_rho_c_all, n_times, n_forward_sim))
            rpi_tightening_per_rho_c = np.zeros(n_rho_c_all)
            for rho_c_idx, rho_c in enumerate(rho_c_all):
                if rho_c_idx < n_rho_c_all - 1:
                    print(f"rho_c iter {rho_c_idx}/{n_rho_c_all - 1}", end="\r")
                else:
                    print(f"rho_c iter {rho_c_idx}/{n_rho_c_all - 1}")
                for t_idx in range(n_times):
                    for k_idx in range(n_forward_sim):
                        w_bar_c_all[rho_c_idx, t_idx, k_idx] = (
                            lyap_err[t_idx, k_idx]
                            * rho_c
                            / (1 - math.exp(-rho_c * (1 + k_idx) * dt_tmpc))
                        )
                rpi_tightening_per_rho_c[rho_c_idx] = np.max(
                    w_bar_c_all[rho_c_idx] / rho_c
                )
            w_bar_c = np.min(rpi_tightening_per_rho_c)
            print(f"{ros_rec_json_name} - w_bar_c: {w_bar_c}")

        # Compute optimal rho_c value
        rho_c_idx = 0
        if compute_rho_c:
            rho_c_idx = np.argmin(rpi_tightening_per_rho_c)
            rho_c = rho_c_all[rho_c_idx]
            print(f"rho_c: {rho_c} at rho_c index: {rho_c_idx}")

        # Compute tube size over time
        s = np.zeros(1 + n_forward_sim)
        for k_idx in range(1 + n_forward_sim):
            s[k_idx] = (1 - math.exp(-rho_c * k_idx * dt_tmpc)) * w_bar_c / rho_c

        # Determine epsilon at all time steps
        epsilon_all = np.zeros(n_tmpc)
        for t in range(n_tmpc):
            t_x_cur_idx = np.abs(t_x_cur - t_x_cur_est[t]).argmin()
            x_err = x_cur[t_x_cur_idx, :] - x_cur_est[t, :]
            epsilon_all[t] = np.sqrt(x_err.T @ P_delta @ x_err)
        epsilon_all = epsilon_all[n_idx_ignore:]
        epsilon = np.max(epsilon_all)
        print(f"{ros_rec_json_name} - epsilon: {epsilon}")

        # Create (rho_c,w_bar_c) figure
        if compute_rho_c:
            if do_plot_lyap_err:
                fig, ax = plt.subplots()
                fig.suptitle(
                    f"{ros_rec_json_name} - Lyapunov error and tube over prediction stages"
                )
                for t_idx in range(n_times):
                    ax.plot(
                        np.arange(1, 1 + n_forward_sim),
                        lyap_err[t_idx],
                    )
                ax.plot(
                    np.arange(1, 1 + n_forward_sim),
                    np.max(lyap_err, axis=0),
                    linewidth=linewidth,
                    label=r"$max(\sqrt{V^\delta(x_{t+\tau},z_{\tau|t})})$",
                )
                ax.plot(
                    np.arange(1 + n_forward_sim),
                    s,
                    linewidth=linewidth,
                    label=r"$s$",
                )
                ax.axhline(
                    y=w_bar_c / rho_c,
                    color="red",
                    linestyle="--",
                    linewidth=linewidth,
                    label=r"$\frac{\bar{w}^\mathrm{c}}{\rho^\mathrm{c}}$",
                )
                ax.set_xlabel(f"Prediction stage k")
                ax.set_ylabel(r"$\sqrt{V^\delta(x_{t+\tau},z_{\tau|t})}$")
                ax.legend()

            if do_plot_rpi_tightening_over_rho_c:
                fig, ax = plt.subplots()
                fig.suptitle(f"{ros_rec_json_name} - RPI tightening vs. rho_c")
                ax.plot(rho_c_all, rpi_tightening_per_rho_c)
                ax.set_xlabel(r"$\rho^\mathrm{c}$")
                ax.set_ylabel(r"$\frac{\bar{w}^\mathrm{c}}{\rho^\mathrm{c}}$")

        # Create w_bar_c figure
        if do_plot_w_bar_c_time:
            fig, ax = plt.subplots()
            fig.suptitle(f"{ros_rec_json_name} - computed w_bar_c over time")
            ax.plot(t_x_cur_est[n_idx_ignore + 1 :], w_bar_c_all[50, :])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(r"$\bar{w}^\mathrm{c}$")

        # Create w_bar_c sorted figure
        if do_plot_w_bar_c_sorted:
            fig, ax = plt.subplots()
            fig.suptitle(f"{ros_rec_json_name} - computed w_bar_c sorted")
            ax.plot(
                np.arange(n_idx_ignore + 1, n_tmpc), sorted(w_bar_c_all[rho_c_idx, :])
            )
            ax.set_xlabel("Index")
            ax.set_ylabel(r"$\bar{w}^\mathrm{c}$")

        # Create epsilon figure
        if do_plot_epsilon_time:
            fig, ax = plt.subplots()
            fig.suptitle(f"{ros_rec_json_name} - computed epsilon over time")
            ax.plot(t_x_cur_est[n_idx_ignore:], epsilon_all)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("$\epsilon$")

        # Create epsilon sorted figure
        if do_plot_epsilon_sorted:
            fig, ax = plt.subplots()
            fig.suptitle(f"{ros_rec_json_name} - computed epsilon sorted")
            ax.plot(np.arange(n_idx_ignore, n_tmpc), sorted(epsilon_all))
            ax.set_xlabel("Index")
            ax.set_ylabel("$\epsilon$")

    plt.show()
