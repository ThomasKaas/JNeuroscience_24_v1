import numpy as np
import pandas as pd

from scipy import signal
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

import heka_reader #External Module by Luke Campagnola to read .dat files from HEKA

from parameters_train_analysis import * #External Script which defines all the parameters needed for further analysis

### Function 1: Helper function, called via multiple functions.  
def load_files_from_mask(age, frequency):
    # Loads Meta-Data from prespecified Excel-Sheet in path_meta_data 
    # Selects all trains under the previously specified categories, e.g. "Age" & "Frequeny"
    # Returns a Dataframe "mask" containing "Filename, Series & Sweeplist" as columns
    mask = (
        pd
        .read_excel(path_meta_data, skiprows=1) #Import from Datenübersicht
        .query('Selected == "x" & Frequency==@frequency & Age==@age') #Select Data at specific Frequency / Age
        .reset_index(drop=True)
    )
    return mask

### Function 2: Helper function, called via multiple functions.  
def import_file_from_batch(mask, i, mode):
    # Extract information from the dataframe "mask"
    bundle = heka_reader.Bundle(path_raw_data + surname + mask['Filename'][i][12:] + '.dat')
    sweeplist = mask['Sweeps for Analysis'][i]
   
    # Convert sweeplist to a list of sweep indices
    if type(sweeplist) == int:
        sweep_indices = [sweeplist]
    else:
        sweep_indices = [int(s) for s in sweeplist.split(',')]

    series_index = mask['Series'][i] - 1  # Series index adjusted for zero-based indexing

    # Process based on the specified mode
    if mode == 'avg':
        # Calculate the average trace
        trace = np.mean([bundle.data[0, series_index, idx - 1, 0] for idx in sweep_indices], axis=0)
    elif mode == 'ind':
        # Extract individual traces
        trace = [bundle.data[0, series_index, idx - 1, 0] for idx in sweep_indices]

    return trace

### Function 3: Helper function, called via multiple functions.  
def remove_artefacts(frequency, trace):
    # Get the stimulation start times based on the frequency
    stimstarttimes = stimstarttimes_dict.get(frequency)

    # Iterate over each stimulation start time
    for i in range(0, len(stimstarttimes)):
        # Iterate over the range of points to be interpolated
        for j in range(0, blank_start + blank_end):
            # Interpolate between points before and after the blank region
            interpolation_line = np.linspace(
                trace[stimstarttimes[i] - blank_start],
                trace[stimstarttimes[i] + blank_end],
                num=blank_start + blank_end
            )
            # Replace the original trace values with the interpolated values
            trace[stimstarttimes[i] - blank_start + j] = interpolation_line[0 + j]

    # Baseline correction by subtracting the mean of the first 1000 points
    trace = trace - trace[0:1000].mean()

    return trace

### Function 4: Helper function, called via multiple functions.
def filter_trace(trace,cutoff_frequency=cutoff_frequency,sampling_frequency=sampling_frequency):
    b, a = signal.butter(4,cutoff_frequency/(0.5*(sampling_frequency/1000**-1)),btype='low', analog=False)
    trace = signal.filtfilt(b, a, trace)
    return trace

### Function 5: Helper function, called via Function 6.
def event_analysis(frequency, trace):
    stimstarttimes = stimstarttimes_dict.get(frequency)
    data_phasic_amplitude, data_tonic_amplitude, events = [], [], []
    
    for i in range(len(stimstarttimes)):
        # Baseline of a Single Event
        baseline_event = trace[stimstarttimes[i] - baseline_start : stimstarttimes[i] - baseline_end]
        amplitude_tonic_component = np.mean(baseline_event)
        data_tonic_amplitude.append(amplitude_tonic_component)

        # Amplitude of a Single Event
        event = trace[stimstarttimes[i] + event_start : stimstarttimes[i] + event_end]
        amplitude_subtracted = np.amin(event) - amplitude_tonic_component
        data_phasic_amplitude.append(amplitude_subtracted)
        events.append(trace[stimstarttimes[i] - baseline_start : stimstarttimes[i] + event_end])

    data_phasic_amplitude = [element * 1e+12 for element in data_phasic_amplitude]  # Converts A to pA
    data_tonic_amplitude = [element * 1e+12 for element in data_tonic_amplitude]  # Converts A to pA

    return data_tonic_amplitude, data_phasic_amplitude, events

### Function 6: Directly called from .ipynb for analysis of synchronous release.
def train_analysis_batched(age, frequency, filtered=True, del_artefacts=True, export=False):
    """
    Function Signature:
        age, frequency: Lists of age and frequency combinations to analyze.
        filtered, del_artefacts, export: Boolean flags for specifying whether to filter traces, delete artifacts, and export results, respectively.

    Initialization:
        Lists traces, events, import_mask, dfs, and dfs_2 are initialized to store trace data, event analysis results, mask information, basic analysis DataFrames, and advanced analysis DataFrames, respectively.

    Loop Over Age and Frequency:
        Nested loops iterate over each combination of age and frequency.

    Load and Preprocess Data:
        Loads mask files based on the current age and frequency.
        For each file in the mask:
            Imports the average trace data.
            Optionally removes artifacts and filters the trace.
            Appends the processed trace to the traces list.

    Event Analysis and Basic Dataframe Creation:
        Runs event analysis (event_analysis) on the trace.
        Creates a basic DataFrame (df) containing information such as filename, age, frequency, number of EPSCs, baseline amplitude, and amplitude.
        Calculates the relative amplitude and adds it to the DataFrame.
        Appends the DataFrame to the dfs list.

    Recovery Curve Fitting and Advanced Dataframe Creation:
        Fits a recovery curve based on the frequency of EPSC.
        Constructs a recovery curve function using the fitted parameters.
        Calculates the steady-state amplitude and recovery time constant (Tau_Recovery).
        Creates an advanced DataFrame (df_advanced) containing information such as filename, age, frequency, 1st EPSC amplitude, paired-pulse ratio, steady state, and Tau_Recovery.
        Appends the advanced DataFrame to the dfs_2 list.

    Concatenate DataFrames:
        Concatenates the basic and advanced DataFrames into df_final and df_advanced_final, respectively.

    Export Data:
        If export is set to True, it exports the basic and advanced DataFrames to Excel files.

    Return Values:
        Returns the lists of imported masks, basic and advanced DataFrames, traces, and events.
    """
    traces, events, import_mask, dfs, dfs_2 = [], [], [], [], []

    # Iterate over age and frequency combinations
    for j in range(len(age)):
        for k in range(len(frequency)):
            # Load mask files based on age and frequency
            mask = load_files_from_mask(age[j], frequency[k])
            import_mask.append(mask)

            # Iterate over files in the mask
            for i in range(len(mask['Filename'])):
                # Import trace from the batch
                trace = import_file_from_batch(mask, i, mode='avg')

                # Remove artifacts if specified
                if del_artefacts:
                    trace = remove_artefacts(frequency[k], trace)

                # Apply trace filtering if specified
                if filtered:
                    trace = filter_trace(trace)

                # Append trace to the list
                traces.append(trace)

                # Run train analysis on the trace
                a, b, c = event_analysis(frequency[k], trace)
                events.append(c)

                # Create DataFrame for basic analysis
                df = pd.DataFrame({
                    "Filename": mask['Filename'][i][12:],
                    "Age": age[j],
                    "Frequency": frequency[k],
                    "#EPSC": 1,
                    "Baseline in pA": a,
                    "Amplitude in pA": b
                })

                # Calculate relative amplitude
                df['Amplitude_relative'] = df['Amplitude in pA'] / df.at[0, 'Amplitude in pA'] * 100

                # Display Number of EPSC
                df["#EPSC"] = df.index

                # Append DataFrame to the list
                dfs.append(df)

                # Fit recovery curve based on frequency
                ss_value_fitting = df["Amplitude_relative"][16:21].mean() - 100
                def func(x, b):
                    return ss_value_fitting * np.exp(-x / b) + 100
                y_sample = df['Amplitude_relative'][22:27] if frequency[k] == 10 else df['Amplitude_relative'][21:27]
                popt, _ = curve_fit(func, recovery_times_10Hz if frequency[k] == 10 else recovery_times, y_sample, p0=[100], bounds=([0], [100000]))

                # Create DataFrame for advanced analysis
                df_advanced = pd.DataFrame({
                    "Filename": mask['Filename'][i][12:],
                    "Age": [df['Age'][0]],
                    "Frequency": df['Frequency'][0],
                    "1st EPSC Amp": df.iloc[0]["Amplitude in pA"],
                    "Paired_Pulse_Ratio": df.iloc[1]["Amplitude in pA"] / df.iloc[0]["Amplitude in pA"],
                    "Steady_State": df['Amplitude_relative'][16:21].mean(),
                    "Tau_Recovery (ms)": popt[0]
                })

                # Append advanced DataFrame to the list
                dfs_2.append(df_advanced)

    # Concatenate DataFrames
    df_final = pd.concat(dfs)
    df_advanced_final = pd.concat(dfs_2)
    traces = np.array(traces,dtype=object)
    
    if export:
        df_final.round(decimals=2).to_excel(export_path+'Results_Trains_basic.xlsx', index=False)
        df_advanced_final.round(decimals=2).to_excel(export_path+'Results_Trains_advanced.xlsx', index=False)

    return import_mask, df_final, df_advanced_final, traces, events

### Function 7: Directly called from .ipynb for analysis of synchronous release.
def summary_stats(df_advanced_final):
    ss = df_advanced_final.groupby(["Age", "Frequency"]).agg(
        Mean_SS=("Steady_State", np.mean),
        SEM_SS=("Steady_State", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        n_SS=("Steady_State", "count")
    )
    df_advanced_final_ppr = df_advanced_final.mask(df_advanced_final['Paired_Pulse_Ratio'] < 0.0)
    ppr = df_advanced_final_ppr.groupby(["Age", "Frequency"]).agg(
        Mean_PPR=("Paired_Pulse_Ratio", np.mean),
        SEM_PPR=("Paired_Pulse_Ratio", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        n_PPR=("Paired_Pulse_Ratio", "count")
    )
    df_advanced_final_tau = df_advanced_final.mask(df_advanced_final['Tau_Recovery (ms)']  > 20000)
    #df_advanced_final_tau = df_advanced_final_tau.mask(df_advanced_final_tau['Tau_Recovery (ms)']  < 100)
    tau = df_advanced_final_tau.groupby(["Age", "Frequency"]).agg(
        Mean_Tau=("Tau_Recovery (ms)", np.mean),
        SEM_Tau=("Tau_Recovery (ms)", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        n_Tau=("Tau_Recovery (ms)", "count")
    )
    result = pd.concat([ss,ppr,tau], axis=1)
    return result

### Function 8: Directly called from .ipynb for analysis of asynchronous release.
def cumulative_charge_transfer_analysis(age_list, frequency=50, eventlist=[0,20], event_length=charge_transfer_window):
    """
    Calculates the cumulative charge transfer of EPSC for a given time from peak.

    Input Parameters:
        age_list: List of age criteria to filter data.
        frequency: Frequency of events.
        eventlist: List of events to analyze charge transfer.
        event_length: Length of the event window for charge transfer analysis.

    Initialization:
        Creates an empty list to store individual result dataframes.
        Sets the start jitter for analysis.

    Loop Over Age Criteria:
        Iterates over each age in the age_list.
        Loads trace data files matching the age and frequency criteria.
        Filters and preprocesses the loaded trace data.

    Loop Over Traces:
        Iterates over each trace in the loaded data.
        Calculates charge transfer over time for each trace and event in eventlist.
        Stores the results in a dataframe and appends it to results_list_cumulative_charge_transfer.

    Output:
        Concatenates all dataframes in results_list_cumulative_charge_transfer.
        Returns the concatenated dataframe.
    """

    results_list_cumumlative_chargetransfer = []
    
    start_jitter = delay_cumulative_chargetransfer_analysis
    
    for age in age_list:
        # Filter the Meta-Datafile to select for traces with corresponding age criteria
        mask = load_files_from_mask(age, frequency)
        # Load pre-processed traces data for subsequent analysis
        traces = [filter_trace(remove_artefacts(frequency, import_file_from_batch(mask, i, mode='avg')))
                  for i in range(len(mask['Filename']))]
        # Iterate over every averaged Train to get relative cumulative charge transfer across time for every cell
        for i in range(0,len(traces)):
            cell_index = [i] * charge_transfer_window
            age_index,frequency_index = [age]*charge_transfer_window, [frequency]*charge_transfer_window
            filename   = [mask['Filename'][i]] * charge_transfer_window
            time_axis  = np.arange(0, charge_transfer_window*dt, dt)
            results_cumcharge = []
            for event in eventlist:
                # Select the stimstartime for the corresponding event
                stimstarttime_ncct = stimstarttimes_dict.get(frequency)[event]
                # Select the first EPSC in the given Train and directly convert it to an numpy array
                trace = np.array(traces[i][stimstarttime_ncct + start_jitter:stimstarttime_ncct + start_jitter + charge_transfer_window])
                # Select Window from first EPSC: negative Peak for lenght corresponding to dt*event_lenght
                trace = traces[i][stimstarttime_ncct+np.argmin(trace):stimstarttime_ncct+np.argmin(trace)+charge_transfer_window]
                # Transform EPSC raw data, to ease cumsum-calculations: transform units to pA and make all values positive
                trace = np.array(trace)*-1*scale_factor
                # Calculate absolute cumulative sum
                chargetransfer_abs = np.cumsum(trace)
                # Calculate relative cumulative sum
                chargetransfer_rel = chargetransfer_abs/chargetransfer_abs.max()
                results_cumcharge.append(chargetransfer_rel)
            # Append the results for cumchargetransfer x time for every cell in a given age category to main variable
            results_list_cumumlative_chargetransfer.append(pd.DataFrame({
                 'Filename': filename,
                 'Age': age_index,
                 'Frequency': frequency_index,
                 'cell_index': cell_index,
                 'time': time_axis,
                 **{f'event_{eventlist[i]}': results_cumcharge[i] for i in range(len(eventlist))}
                 }))
    return pd.concat(results_list_cumumlative_chargetransfer, ignore_index=True)

### Function 8.1: Directly called from .ipynb for analysis of asynchronous release.
def CCT_uzay(ages, frequencies, event_duration, eventlist=[0], stim_duration=10): 
    """
    Calculates the cumulative charge transfer of EPSC for a given time from end of stimulation.
    Analysis adapted from Uzay et al. 2023 (Cell Reports)

    Input Parameters:
        ages: List of age criteria.
        frequencies: List of frequencies.
        event_duration: Duration of events.
        eventlist: List of events to analyze charge transfer.
        stim_duration: Duration of stimulus.

    Initialization:
        Sets up result dataframe columns.
        Initializes an empty dictionary to store results.

    Loop Over Age and Frequency:
        Iterates over each age and frequency combination.
        Loads trace data files matching the criteria.
        Processes each trace to calculate charge transfer and IC50 values.

    Output:
        Returns a dictionary containing results for each trace, including filename, age, frequency, event index, event data, charge transfer, and IC50 values.
    """
    # Set structure and naming for columns in the results dict
    result_df_columns = ['filename','age','frequency','event_index' ,'event','cct','ic50_ms']
    # Initialize an empty dictionary with default values as empty lists
    data = {column: [] for column in result_df_columns}
    # Define critical general parameters for analysis
    start_jitter    = stim_duration #estimate of the end of the stimulus artefact in a filtered trace
    cct_time_window = int(event_duration/dt) #time length of window for charge transfer  
    for age in ages:
        for frequency in frequencies:
            mask   = load_files_from_mask(age, frequency)
            # Calculate average trace for every cell matching the selection criteria
            traces = [import_file_from_batch(mask, i, mode='avg') for i in range(len(mask['Filename']))]
            # Calculate  CCT & IC50 values for every trace of a given category matching agexfrequency
            for i in range(0,len(traces)):
                # Iterate over every indexed Event, currently preset to the first event in a given train
                for event_index in eventlist:
                    # Select the stimstartime for the corresponding event
                    stimstarttime_ncct = stimstarttimes_dict.get(frequency)[event_index]
                    # Baseline correction by subtracting the median of the first 1000 sampling points
                    trace = traces[i]-np.median(traces[i][0:1000])
                    # Crop trace to the first EPSC in the given Train and directly convert it to an numpy array
                    trace = np.array(trace[stimstarttime_ncct + start_jitter:stimstarttime_ncct + start_jitter + cct_time_window])
                    # Transform EPSC raw data, to ease cumsum-calculations: transform units to pA and make all values positive
                    event = trace*scale_factor
                    # Calculate absolute cumulative sum
                    chargetransfer_abs = np.cumsum(trace)
                    # Calculate relative cumulative sum
                    chargetransfer_rel = chargetransfer_abs/chargetransfer_abs.max()
                    index_ic50 = np.argmax(chargetransfer_rel > 0.5)
                    ic_50ms = index_ic50*dt
                    data['filename'].append(mask['Filename'][i])
                    data['age'].append(mask['Age'][i])
                    data['frequency'].append(mask['Frequency'][i])
                    data['event_index'].append(event_index)
                    data['event'].append(event)
                    data['cct'].append(chargetransfer_rel)
                    data['ic50_ms'].append(round(ic_50ms,2))
    return data

### Function 9: Directly called from .ipynb for analysis of asynchronous release.
def latency_jitter_analysis(frequencies, ages):
    # Define the columns for the resulting DataFrame
    result_df_columns = ['Filename', 'Age', 'Frequency', 'Series', 'Sweep', 'Event','Amplitude (pA)', 'Delay (ms)']
    # Initialize an empty dictionary with default values as empty lists
    data = {column: [] for column in result_df_columns}
    # Loop over each age and frequency combination
    for age in ages:
        for frequency in frequencies:
            # Load files based on the mask for the current age and frequency
            mask = load_files_from_mask(age, frequency)
            # Remove rows where 'Sweeps for Analysis' column values are integers
            mask = mask[~mask['Sweeps for Analysis'].apply(lambda x: isinstance(x, int))]
            # Split 'Sweeps for Analysis' values into lists of integers
            for i in mask['Sweeps for Analysis'].index:
                mask['Sweeps for Analysis'][i] = [int(s) for s in mask['Sweeps for Analysis'][i].split(',')]
            # Filter rows based on the length of 'Sweeps for Analysis' lists
            mask = mask[mask['Sweeps for Analysis'].apply(lambda x: len(x) >= 3)]
            mask = mask.reset_index(drop=True)
            # Iterate over each File in the filtered mask DataFrame
            for m in range(0, len(mask['Filename'])):
                # Read data bundle from a file
                bundle = heka_reader.Bundle(path_raw_data + surname + mask['Filename'][m][12:] + '.dat')
                # Get the list of sweeps to be analyzed
                sweeplist = mask['Sweeps for Analysis'][m]
                # Iterate over each sweep in the sweeplist
                for sweep_number in range(0, len(sweeplist)):
                    # Extract the sweep data
                    sweep = bundle.data[0, mask['Series'][m] - 1, sweeplist[sweep_number] - 1, 0]
                    # Perform artifact removal and trace filtering
                    sweep = remove_artefacts(frequency, sweep)
                    sweep = filter_trace(sweep)
                    # Define a dictionary mapping frequencies to their corresponding stimulation times
                    frequency_options = {10: stimstarttimes_10, 50: stimstarttimes_50, 100: stimstarttimes_100}
                    # Iterate over events based on the specified frequency
                    for event_number in range(0, len(frequency_options.get(frequency, None)[0:21])):
                        k = frequency_options.get(frequency, None)[event_number]
                        cropped_event = sweep[k:k + 500]
                        # Calculate jitter and append values to the data dictionary
                        delay = (np.argmin(cropped_event)) * dt
                        amplitude = round(np.min(cropped_event)*10e11,2)
                        values = [mask['Filename'][m], age, frequency, mask['Series'][m], sweep_number + 1, event_number + 1,amplitude, delay]
                        # Update the data dictionary
                        data.update({category: data.get(category, []) + [val] for category, val in zip(result_df_columns, values)})
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)
    # Return the resulting DataFrame
    return df

# Function 10: Shall be used in conjunction with the output dataframe of Function 8; when using Function 8.1 Function 10 is not necessary 
def get_ic50_values(df,agelist,eventnumber):
    event = 'event_'+str(eventnumber)
    results = {'Filename':[], 'Age':[],'cell_index':[],'IC_50(ms)_'+event:[]}
    for age in agelist:
        for cell in df.query('Age =='+str(age))['cell_index'].unique():
            # Built query command to filter the dataframe
            query_string = 'Age =='+str(age)+' & cell_index =='+str(cell)
            # Isolate the first Event for the first cell throught matching the query criteria
            series     = df.query(query_string)[event]
            # Define the treshold for the IC50 calculations: 50% of the maximum, equaling to 50%
            threshold  = 0.5*df.query(query_string)[event].max()
            # Update the results dictionary: Get the time, where relative cumulative charge transfer reaches 50% 
            results['IC_50(ms)_'+event].append(round(df.query(query_string)['time'][(series > threshold).idxmax()],3))
            results['Filename'].append(df.query(query_string)['Filename'].unique()[0])
            results['Age'].append(age)
            results['cell_index'].append(cell)
    return pd.DataFrame(results)

# Function 11: Is intended to be performed on the output of Function 9
def get_std_latency(agelist,frequencies,data):
    results = {'Filename':[], 'Age':[],'Frequency':[],'Event':[],'std_delay_(ms)':[]}
    eventlist=[i for i in range(1,22)]
    for age in agelist:
        for frequency in frequencies:
            for event in eventlist:
                querystring   = 'Age == '+str(age)+' & Frequency == '+str(frequency)+' & Event == '+str(event)
                new_df        = data.query(querystring)
                new_df['Filename_Index'] = pd.factorize(new_df['Filename'])[0]
                for i in new_df['Filename_Index'].unique():
                    std = new_df.query('Filename_Index =='+str(i))['Delay (ms)'].std()
                    filename = new_df.query('Filename_Index =='+str(i))['Filename'].unique()[0]
                    results['std_delay_(ms)'].append(std)
                    results['Filename'].append(filename)
                    results['Event'].append(event)
                    results['Age'].append(age)
                    results['Frequency'].append(frequency)
    return pd.DataFrame(results)

# Function 12: Helper Function. Defines the monoexponential Fit to that part of the EPSC
def monoexponential(t, A, tau):
    return A * np.exp(-t / tau)

# Function 13: Directly called from .ipynb for analysis of asynchronous release.
def decay_fit(ages,frequencies,event_number):
    """
    Initialization:
        Two dictionaries, results and data, are initialized to store the fitting results and fitted data, respectively.
        results will contain information such as filename, age, frequency, decay time constant (tau), and R-squared value.
        data will store filename, age, frequency, original EPSC data, and the fitted data for further analysis.

    Loop Over Age and Frequency:
        The function iterates over combinations of ages and frequencies provided as input parameters.

    Data Preparation:
        For each combination of age and frequency, the function loads trace data files matching the criteria.
        It preprocesses the trace data by removing artifacts and filtering it.
        Then, it iterates over each averaged trace obtained from the loaded data.

    Exponential Decay Fitting:
        For each trace, it identifies the negative peak of the EPSC, slices the data from this peak onwards, and defines it as the region of interest for fitting.
        It calculates the time axis (x_data) and the EPSC values (y_data) within this region.
        Initial parameter guesses for the exponential decay fitting are determined based on the minimum and maximum values of the y_data.
        The curve_fit function from scipy.optimize module is used to perform monoexponential fitting using the monoexponential function.
        The fitted data is obtained using the parameters obtained from the fitting process.
        The decay time constant (tau) and R-squared value of the fitting are calculated.

    Results Collection:
        The obtained fitting results (tau and R-squared) along with metadata (filename, age, and frequency) are stored in the results dictionary.
        Additionally, the original EPSC data (y_data) and the fitted data (y_data_fitted) are stored in the data dictionary for quality check and further analysis.

    Output:
        Finally, the function returns two data structures: a Pandas DataFrame containing the fitting results (results) and a dictionary containing original and fitted data (data) for each trace.
    """
    results = {'Filename':[], 'Age':[],'Frequency':[],'tau (ms)':[], 'r^2':[]}
    data = {'Filename':[], 'Age':[],'Frequency':[], 'y_data':[],'y_data_fitted':[]}
    for age in ages:
        for frequency in frequencies:
            # Create List of all the traces that qualify for given age & frequency
            mask   = load_files_from_mask(age, frequency)
            traces = [filter_trace(remove_artefacts(frequency, import_file_from_batch(mask, i, mode='avg')))
              for i in range(len(mask['Filename']))]
            
            # Iterate over every averaged trace
            for j in range(0,len(traces)):
                trace_index = j
                
                # Create New Array which is a sliced version such that it starts with the negative Peak
                epsc_index = stimstarttimes_dict[frequency][event_number] # 0 means first EPSC
                epsc = traces[trace_index][epsc_index:epsc_index + decay_interval]
                negative_peak_index = np.argmin(epsc)
                cropped_epsc = epsc[negative_peak_index:]

                # Crop ydata and create time axis as xdata
                index_80_perc_peak = np.abs(cropped_epsc - (perc_neg_peak*epsc[negative_peak_index])).argmin()
                y_data = cropped_epsc[index_80_perc_peak:]
                x_data = np.linspace(0, len(y_data)*dt, len(y_data))

                # Perform Fit
                amplitude_guess = np.min(y_data)-np.max(y_data)
                delay_guess = x_data[np.argmax(y_data)]
                popt, _ = curve_fit(monoexponential, x_data, y_data, p0=(amplitude_guess,delay_guess))
                fitted_data = monoexponential(x_data, popt[0], popt[1])
                tau = round(popt[1], 2)
                r_squared = round(r2_score(y_data, fitted_data),3)
                
                # Collect results and store it in the previously initialized dictionary
                results['tau (ms)'].append(tau)
                results['r^2'].append(r_squared)
                results['Filename'].append(mask['Filename'][j])
                results['Age'].append(age)
                results['Frequency'].append(frequency)
                
                # Store Data from fitting procedure to create plots for quality check afterwards
                data['Filename'].append(mask['Filename'][j])
                data['Age'].append(age)
                data['Frequency'].append(frequency)
                data['y_data'].append(y_data*10e11)
                data['y_data_fitted'].append(fitted_data*10e11)
    return pd.DataFrame(results),data