import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import time

class rt_filtering:
    """
    Class for real-time filtering of EMG data
    """
    def __init__(self, sample_rate, lp_cutoff=3, bandpass_high=450, bandpass_low=20, order=2):
        """
        Initializes the filters for real-time processing of EMG data.
        Parameters:
        - sample_rate: Sampling rate of the EMG data in Hz
        - lp_cutoff: Low-pass filter cutoff frequency in Hz (default: 3)
        - bandpass_high: High-frequency cutoff for bandpass filter in Hz (default: 450)
        - bandpass_low: Low-frequency cutoff for bandpass filter in Hz (default: 20)
        - order: Filter order (default: 2)
        """
        self.fs = sample_rate       # Sample rate in Hz
        self.nyq = 0.5 * self.fs    # Nyquist frequency

        # --- design filters (SOS) ---
        self.lp_sos = butter(2, lp_cutoff / self.nyq, btype='lowpass', output='sos')
        self.lp_zi  = sosfilt_zi(self.lp_sos) * 0.0 
        
        self.hp_sos = butter(order, bandpass_low / self.nyq, btype='highpass', output='sos')
        self.hp_zi  = sosfilt_zi(self.hp_sos) * 0.0

        self.bandpass_sos = butter(order, [bandpass_low / self.nyq, bandpass_high / self.nyq], btype='bandpass', output='sos')
        self.bandpass_zi  = sosfilt_zi(self.bandpass_sos) * 0.0
    
    def RMS(self, window, window_size=50):
        """
        Calculates the Root Mean Square (RMS) of a given window of data.
        Parameters:
        - window: Array of data values for which to calculate the RMS
        - window_size: Size of the window to consider for RMS calculation (default: 50
        Returns:
        - RMS value of the given window
        """
        if len(window) < window_size:
            return window[-1]
        return float(np.sqrt(np.mean(np.square(np.abs(window)))))
    
    def bandpass(self, data):
        """
        Applies a bandpass filter to the input data.
        Parameters:
        - data: input data to be filtered
        Returns:
        - Filtered data after applying the bandpass filter
        """
        y, self.bandpass_zi = sosfilt(self.bandpass_sos, data, zi=self.bandpass_zi)
        return y
    
    def lowpass(self, data):
        """
        Applies a low-pass filter to the input data.
        Parameters:
        - data: input data to be filtered
        Returns:
        - Filtered data after applying the low-pass filter
        """
        y, self.lp_zi = sosfilt(self.lp_sos, data, zi=self.lp_zi)
        return y

class rt_net_muscle_activation_lowpass:
    """
    Class for real-time low-pass filtering of the net muscle activation for trajectory generation
    """
    def __init__(self, sample_rate, lp_cutoff=2, order=2):
        """
        Initializes the low-pass filter for real-time processing of net muscle activation data.
        Parameters:
        - sample_rate: Sampling rate of the net muscle activation data in Hz
        - lp_cutoff: Low-pass filter cutoff frequency in Hz (default: 2)
        - order: Filter order (default: 2)
        """
        self.fs = sample_rate       # Sample rate in Hz
        self.nyq = 0.5 * self.fs    # Nyquist frequency

        # --- design filters (SOS) ---
        self.lp_sos = butter(order, lp_cutoff / self.nyq, btype='lowpass', output='sos')
        self.lp_zi  = sosfilt_zi(self.lp_sos) * 0.0

    def lowpass(self, data):
        """
        Applies a low-pass filter to the input data.
        Parameters:
        - data: input data to be filtered
        Returns:
        - Filtered data after applying the low-pass filter
        """
        y, self.lp_zi = sosfilt(self.lp_sos, data, zi=self.lp_zi)
        return y

    def reset(self):
        """
        Resets the filter state to zero. This can be useful when starting a new trajectory or when the filter state needs to be cleared.
        """
        self.lp_zi  = sosfilt_zi(self.lp_sos) * 0.0