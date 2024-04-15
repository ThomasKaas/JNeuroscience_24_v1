import sys, os, time
import numpy as np

timestr              = time.strftime("%Y%m%d-%H%M%S") #For Version Control
### Import
path_raw_data        = 'data_example/'
path_meta_data       = 'data_example/meta_data_file_example.xlsx'
### Export
export_path          = 'output/'+timestr
surname              = 'Thomas Kaas_'
export_folder_events = 'output/'+timestr+'/Events'


### Specify General Parameters of the recorded Data
dt = 0.02 # Sampling interval in ms
sampling_frequency = int(1/dt)
scale_factor = 1e+12 ##Scale Factor for EPSC Amplitudes in pA

# Stimulation Times (quantified as absolut number of data point) used for the Train Recording:
stimstarttimes_10  = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 113750, 120000, 132500, 157500, 207500, 357500]
stimstarttimes_50  = [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 33750, 40000, 52500, 77500, 127500, 277500]
stimstarttimes_100 = [10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 23750, 30000, 42500, 67500, 117500, 267500]

#Stimulation Times compressed into a dictionary, that is used throughout the following functions to compress code
stimstarttimes_dict = {10: stimstarttimes_10, 50: stimstarttimes_50, 100: stimstarttimes_100}

# Recovery Times (quantified as ms after last stimulus in the train sequence)
recovery_times      = np.array([75,200,450,950,1950,4950]) #absolute times, dt is increasing by factor 2
recovery_times_10Hz = np.array([200,450,950,1950,4950])    #left 75ms because <100ms ISI during the train
###Specify Window for Event Analysis
baseline_start = 100 # Starts 100 Sample points before Onset of Stimulation
baseline_end   = 20  # Ends 20 Sample points before Onset of Stimulation
event_start    = 50  # Starts 50 Sample points after Onset of Stimulation
event_end      = 500 # Ends 500 Sample points after Onset of Stimulation
blank_start    = 30  # Number of Data Points before Onset of Stim Artefact for Interpolation Start
blank_end      = 60  # Number of Data Points after Onset of Stim Artefact for Interpolation End


cutoff_frequency = 5000 # Set Lowpass Filter cutoff frequency

### PARAMETERS FOR ANALYSIS OF ASYNCHRONOUS RELEASE
charge_transfer_window = 500 #defines the number of sampling points over which the charge is integrated
delay_cumulative_chargetransfer_analysis = 50 #defines the addidional number of sampling points after which cumuluate charge transfer analysis is started
event_number_start = 0  # first cumulative charge transfer is calculated for the first event in the train
event_number_end   = 20 # second cumulative charge transfer is calculated for the last event in the train

decay_interval = 500 #corresponds to 10ms
perc_neg_peak  = 0.8 #corresponds to the percentage of the negative amplitude after which to start the fitting procedure
