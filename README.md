# mpc_model_id_mismatch

This repository provides a package with the following scripts:

- [rosbag_to_json.py](scripts/rosbag_to_json.py): to convert rosbag files to json files that can be used in the subsequent scripts
- [setup_venv.sh](setup_venv.sh): to create a Python virtual environment with all required dependencies to run the subsequent scripts
- [determine_model_coefficients.py](scripts/determine_model_coefficients.py): to determine the coefficients of a nonlinear system model via a filtering-based technique
- [determine_model_mismatch.py](scripts/determine_model_mismatch.py): to determine the model mismatch of this model with respect to measured simulation or experimental data
- [plot_model_mismatch_data.py](scripts/plot_model_mismatch_data.py): to plot the model mismatch results

## Install

To run [rosbag_to_json.py](scripts/rosbag_to_json.py), we use the following packages globally on the system:

- [ROS Noetic](https://wiki.ros.org/noetic) (custom message [DroneFalconOutput](../simple_sim/msg/DroneFalconOutput.msg) needs to be known by building the [simple_sim](../simple_sim) package)
- [Python 3.8.10](https://www.python.org/downloads/release/python-3810/)

To run [determine_model_mismatch.py](scripts/determine_model_mismatch.py), you need to have the following additional packages installed:

- [ACADOS release v0.3.5](https://github.com/acados/acados/releases/tag/v0.3.5). Install this package at a convenient location on your system.

## Run

1. Adjust the settings in [rosbag_to_json.yaml](config/scripts/rosbag_to_json.yaml) and convert the rosbag file to a json file using the script [rosbag_to_json.py](scripts/rosbag_to_json.py). We recommend to run this script using VS Code by selecting the correct Python interpreter and pressing CTRL+F5.

   This script will create json files in the [data/converted_bags](data/converted_bags) folder. If these files already exist, you do not have to run this conversion script. Otherwise, you will overwrite the existing files.

2. Create the Python virtual environment. This environment contains all required dependencies you need to run the subequent scripts. Create the environment by making sure that _setup_venv.sh_ is executable:

   ```bash
   chmod +x setup_venv.sh
   ```

   and by running the script:

   ```bash
   setup_venv.sh [/path/to/acados_template]
   ```

   The script will create a directory called _venv_ in the root of this repository.

   > :bulb: The _path/to/acados_template_ is the path to the directory _path/to/acados/interfaces/acados_template_. This acados Python package needs to be found to generate the solver to solve the receding horizon optimization problem used in [determine_model_mismatch.py](scripts/determine_model_mismatch.py). It is an optional argument. However, if not specified, you will not be able to run the script successfully.

3. Activate the Python virtual environment by running the following command in a terminal:

   ```bash
   source venv/bin/activate
   ```

4. Adjust the settings in [determine_model_coefficients.yaml](config/scripts/determine_model_coefficients.yaml) and run the script [determine_model_coefficients.py](scripts/determine_model_coefficients.py) to determine the coefficients of the nonlinear system model:

   ```bash
   python scripts/determine_model_coefficients.py
   ```

   The script will estimate model coefficients of which the name is included in the file name and print the desired model coefficient values that can be used in one of the _yaml_ files in the [config/systems](config/systems) folder.

5. Adjust the settings in [determine_model_mismatch.yaml](config/scripts/determine_model_mismatch.yaml) and run the script [determine_model_mismatch.py](scripts/determine_model_mismatch.py) to determine the model mismatch of the nonlinear system model with respect to measured simulation or experimental data:

   ```bash
   python scripts/determine_model_mismatch.py
   ```

   The script will create a _json_ and a _mat_ file in the [data/model_mismatch_results](data/model_mismatch_results) folder. The _json_ file contains all data needed to plot the model mismatch results in the next step. The _mat_ file contains the data required in possible subsequent steps to compute a control law and corresponding safe set for the system.

6. Plot the model mismatch results using the script [plot_model_mismatch_data.py](scripts/plot_model_mismatch_data.py):

   ```bash
   python scripts/plot_model_mismatch_data.py
   ```

   The script creates plots from which you can visually inspect the model mismatch and related quantities.
