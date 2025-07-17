import numpy as np
import yaml

from mpc_model_id_mismatch import helpers
from pathlib import Path
from os import path

if __name__ == "__main__":
    # Print settings
    np.set_printoptions(linewidth=np.inf, precision=10)

    # Load tmp_tio_data.json
    package_dir = Path(__file__).parents[1]
    config_dir = f"{package_dir}/config"
    config_path = f"{config_dir}/scripts/rk4_test.yaml"
    data_dir = f"{package_dir}/data"

    # Read configuration parameters
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Get common parameters
    quad_name = config["model"]["name"]
    g = config["constants"]["g"]

    # Create model
    params_file = f"{config_dir}/systems/{quad_name}.yaml"
    if path.exists(params_file):
        print(f"Selected {quad_name} params file")
    else:
        print(f"Unknown quad name {quad_name}! Exiting")
        exit(1)
    if quad_name == "falcon":
        model = helpers.DroneAgiModel(quad_name, g, params_file)
    else:
        print(f"Unknown model {quad_name}! Exiting")
        exit(1)

    # Set input and state to forward-simulate using RK4
    # u = np.array([1.517866927, 1.517625838, 1.517866833, 1.517625778])
    # x = np.array(
    #     [
    #         -1.088039461e-07,
    #         0.04513214312,
    #         1.005906148,
    #         -0.003056814527,
    #         4.019723667e-07,
    #         3.111125679e-08,
    #         -6.970062469e-08,
    #         0.08404872424,
    #         -0.01920001402,
    #         0.008827651442,
    #         -1.662763141e-06,
    #         2.035523182e-07,
    #     ]
    # )
    # u = np.array([1.513484578, 1.512785285, 1.513484578, 1.512785275])
    # x = np.array(
    #     [
    #         7.13396327e-09,
    #         0.005192631947,
    #         1.021236713,
    #         -0.01448638414,
    #         -2.632632213e-08,
    #         -3.367524205e-09,
    #         2.946288658e-09,
    #         0.03863876329,
    #         -0.02817144701,
    #         0.03214098792,
    #         1.216638564e-07,
    #         -3.168494231e-08,
    #     ]
    # )
    u = np.array([1.515089145, 1.514529085, 1.515089106, 1.514529055])
    x = np.array(
        [
            6.357276114e-09,
            0.009717084011,
            1.018401242,
            -0.01159208625,
            -1.191820659e-08,
            -7.074713812e-09,
            -1.611965792e-08,
            0.05137711841,
            -0.02853182369,
            0.02574496998,
            1.674338067e-07,
            -3.873312293e-08,
        ]
    )

    # Set sampling time
    dt = 0.01

    # Set number of steps
    n_times = 11

    # Forward-simulate using RK4
    n_u = u.shape[0]
    n_x = x.shape[0]
    x_rk4 = np.zeros((n_x, n_times))
    x_rk4[:, 0] = x
    for i in range(1, n_times):
        x_rk4[:, i] = np.array(
            helpers.solve_rk4(model.state_update_ct, x_rk4[:, i - 1], u, dt)
        ).flatten()

    # Print resulting state
    print(f"x_rk4:\n{x_rk4}")
