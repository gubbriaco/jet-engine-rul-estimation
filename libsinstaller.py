import subprocess
import sys
import logging

class Installer():
    """
    A class to manage the installation of Python libraries using pip, with logging support
    for tracking each installation attempt, successful operations, and error details.

    Methods:
        __init__(): Initializes the Installer class, setting up logging configuration.
        _run_pip_command(command): Executes a pip command and logs its execution status.
        _upgrade_pip(): Upgrades pip to the latest version, logging the result.
        install(libraries): Installs a list of libraries and logs the progress and outcomes.
    """
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
            handlers=[
                logging.FileHandler('libraries.log'),  # Save logs to a file
                logging.StreamHandler(sys.stdout)  # Also print logs to the console
            ]
        )

    def _run_pip_command(self, command):
        try:
            logging.info(f"Running pip command: {command}")
            subprocess.check_call([sys.executable, "-m", "pip"] + command.split())
            logging.info(f"Command '{command}' executed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while executing command '{command}'. Error: {e}")
            raise

    def _upgrade_pip(self):
        try:
            logging.info('Upgrading pip...')
            self._run_pip_command('install --upgrade pip')
            logging.info("pip upgraded successfully.")
        except Exception as e:
            logging.error(f"Failed to upgrade pip: {e}")
            raise

    def install(self, libraries):
        """
        Installs a list of Python libraries specified by `libraries`.
        
        Args:
            libraries (list): A list of strings, where each string is the name of a library to install.
        
        Raises:
            Exception: If pip upgrade or any library installation fails, an error is logged, and the exception is raised.
        
        Example:
            installer = Installer()
            installer.install(['requests', 'numpy'])
        """
        self._upgrade_pip()
        for library in libraries:
            try:
                logging.info(f"Installing library {library}...")
                self._run_pip_command(f"install {library}")
                logging.info(f"{library} installed successfully.")
            except Exception as e:
                logging.error(f"Failed to install {library}: {e}")
        logging.info('Installation of all libraries completed!')


libs_installer = Installer()
libs_installer.install(
    libraries=[
        'numpy',
        'pandas',
        'tqdm',
        'gdown',
        'joblib',
        'pickle',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tensorflow',
        'keras-tuner',
        'hiplot'
    ]
)