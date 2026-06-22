# Intention Based Trajectory Generation
This repository contains the code for the paper "Intention based trajectory generation for control of upper limb exosuit" by Victor Brønsholm Nielsen and Xiaofeng Xiong.

# Content
This repository contains the following:
- Calibration
- > ApplyCalibration.py
  Contains the code to normalize processed EMG signals using either MVC or 95th percentile normalization.
  > MVCCalibration.py
  Contains the code to perform a MVC calibration.
  > PercentileCalib.py
  Contains the code to perform a 95th percentile calibration.

- Methods
- > DirectMappingMethods.py
  Contains the code for Method 1 and Method 2 decribed in the paper.
  > IntegrationBasedMethods.py
  Contains the code for Method 3-10 described in the paper.
  > pDMPMethods.py
  Contains the code for Method 11 and Method 12 described in the paper.

- Motors
- > DynamixelHardwareInterface.py
  Contains code to inteface with the Dynamixel motors.

- PredictionModels
- > ESN.py
  Contains code for the ESN model.
  > ESNRealDataTrain.py
  Contains the code used to train and evaluate the ESN prediction model on real world recorded data.
  > ESNTrain.py
  Contains code to train and evaluate the ESN network on simulated data.
  > LSTM.py
  Contains the code for the LSTM model.
  > LSTMRealDataTrain.py
  Contains the code used to train and evaluate the LSTM prediction model on real world recorded data.
  > LSTMTrain.py
  Contains code to train and evaluate the LSTM network on simulated data.
  
- Sensors
- > Pytrigno
  Contains a modified version of the PyTrigno library.
  > DelsysAPIStream.py
  Trigno SDK TCP client that mimics the pytrigno-based API.
  > EMGSensors.py
  Contains the code used to read EMG and IMU data from the EMG sensors using the modified PyTrigno library.

- SignalProcessing
- > Filtering.py
  Contains all the filters used to process the EMG signal.

- SimulationOfEMGMethods.py
Contains code that tests each trajectory generation method on a simulated input signal.

- main.py
Contains the code used to test the full framework with a PID controller for the evaluation in the paper.

- Requiremnts.txt
Contians standard Python library dependencies.

- Requirements_PyTorch.txt
Contains the PyTorch dependencies.


# Dependencies
This repository requires the Delsys Trigno software containing the Trigno Control Utility which can be found here: https://delsys.com/software/
This repository further depends on the Dynamixel SDK, and it is also recommended to download the Dynamixel wizzard.

The standard Python Library dependencies can be found in Requirements.txt, and they can be downloaded directly using the following command:
- pip install -r Requirements.txt

The PyTorch dependencies can be found in Requirements_PyTorch.txt and can be downloaded using the following command:
- pip install -r Requirements_PyTorch.txt
or alternatively it can be downloaded using the interactive PyTorch get started page here:
https://pytorch.org/get-started/locally/
