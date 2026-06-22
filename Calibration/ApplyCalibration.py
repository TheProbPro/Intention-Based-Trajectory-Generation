import numpy as np
import pandas as pd
import os

class ApplyCalibration:
    def __init__(self, user_name='User', Calibration='percentile', BicepEMG = True, TricepEMG = True):
        """
        Initialize the ApplyCalibration class.
        :param user_name: Name of the user for calibration data storage
        :param Calibration: Type of calibration to use (either 'percentile' or 'MVC')
        :param BicepEMG: Boolean to indicate if bicep EMG is used
        :param TricepEMG: Boolean to indicate if tricep EMG is used
        """
        self.user_name = user_name
        self.Calibration = Calibration
        self.Bicep_EMG = BicepEMG
        self.Tricep_EMG = TricepEMG

        # Check if there is a user with given name
        if not os.path.exists(f'Calib/Users/{self.user_name}'):
            print("Please run the calibration script for the user before starting the program!")
            raise ValueError("User not found!")

        # Load users biscep and tricep rest signal from .csv file
        if self.Calibration == 'percentile':
            if not os.path.exists(f'Calib/Users/{self.user_name}/percentile/rest_signal.csv'):
                print("Please run the calibration script for the user before starting the program!")
                raise ValueError("Rest signal not found!")
            df = pd.read_csv(f'Calib/Users/{self.user_name}/percentile/rest_signal.csv')
            if BicepEMG:
                self.bicep_rest = float(df['Bicep'])
            if TricepEMG:
                self.tricep_rest = float(df['Tricep'])
        elif self.Calibration == 'MVC':
            if not os.path.exists(f'Calib/Users/{self.user_name}/MVC/rest_signal.csv'):
                print("Please run the calibration script for the user before starting the program!")
                raise ValueError("Rest signal not found!")
            df = pd.read_csv(f'Calib/Users/{self.user_name}/MVC/rest_signal.csv')
            if BicepEMG:
                self.bicep_rest = float(df['Bicep'])
            if TricepEMG:
                self.tricep_rest = float(df['Tricep'])
        
        # Load users biscep and tricep max signal from .csv file
        if self.Calibration == 'percentile':
            if not os.path.exists(f'Calib/Users/{self.user_name}/percentile/max_signal.csv'):
                print("Please run the calibration script for the user before starting the program!")
                raise ValueError("Max signal not found!")
            df = pd.read_csv(f'Calib/Users/{self.user_name}/percentile/max_signal.csv')
            if BicepEMG:
                self.bicep_max = float(df['Bicep'])
            if TricepEMG:
                self.tricep_max = float(df['Tricep'])
        elif self.Calibration == 'MVC':
            if not os.path.exists(f'Calib/Users/{self.user_name}/MVC/max_signal.csv'):
                print("Please run the calibration script for the user before starting the program!")
                raise ValueError("Max signal not found!")
            df = pd.read_csv(f'Calib/Users/{self.user_name}/MVC/max_signal.csv')
            if BicepEMG:
                self.bicep_max = float(df['Bicep'])
            if TricepEMG:
                self.tricep_max = float(df['Tricep'])

    def compute_activation(self, env):
        """
        Computes mucle wise activation based on the calibration data
        :param env: The input EMG signal (can be a single value or a list/array of [bicep, tricep])
        :return: Tuple of (bicep_activation, tricep_activation) where each activation is between 0 and 1
        """
        if not isinstance(env, (np.ndarray, list)):
            if self.Bicep_EMG:
                env = [env, 0.0]
            elif self.Tricep_EMG:
                env = [0.0, env]
            else:
                raise ValueError("At least BicepEMG or TricepEMG must be True")
        a_bicep = 0
        a_tricep = 0
        if self.Bicep_EMG:
            a_bicep = np.clip(((env[0] - self.bicep_rest) / (self.bicep_max - self.bicep_rest)), 0, 1)
        if self.Tricep_EMG:
            a_tricep = np.clip(((env[1] - self.tricep_rest) / (self.tricep_max - self.tricep_rest)), 0, 1)

        return a_bicep, a_tricep