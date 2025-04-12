# How to create a new environment

## Using the `Anaconda Distribution`

1. Download and install the Anaconda Distribution from the [official website](https://www.anaconda.com/download).
2. Open the Anaconda Navigator.
3. Click on the `Environments` tab.
4. Click on the `Create` button.
5. Enter the name of the environment and select the Python version.
6. Click on the `Create` button.
7. The new environment will be created.

## Using the default `venv` module

#### Windows (Command Prompt)

1. Open the Command Prompt.
2. Navigate to the directory where you want to create the environment.
3. Run the following command to create a new environment: `python -m venv myenv`
4. Activate the environment by running the following command: `myenv\Scripts\activate`
5. The environment is now activated.

#### Windows (git bash)

1. Open the git bash terminal.
2. Navigate to the directory where you want to create the environment.
3. Run the following command to create a new environment: `python -m venv myenv`
4. Activate the environment by running the following command: `source myenv/Scripts/activate`
5. The environment is now activated.

#### MacOS/Linux

1. Open the terminal.
2. Navigate to the directory where you want to create the environment.
3. Run the following command to create a new environment: `python3 -m venv myenv`
4. Activate the environment by running the following command: `source myenv/bin/activate`
5. The environment is now activated.

## Notes

- Replace `myenv` with the desired name of the environment.
- After activating the environment, you should see the environment name in the terminal prompt.
- To deactivate the environment, run the `deactivate` command in the terminal.
- You can install packages in the environment using `pip install package_name`.
- You can create a `requirements.txt` file by running `pip freeze > requirements.txt` to save the installed packages.
- You can install packages from a `requirements.txt` file by running `pip install -r requirements.txt`.