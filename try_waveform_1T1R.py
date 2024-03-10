#####################################################################
## Import functions
#####################################################################
#%%
## Communication with setup
from cassini.system import CassiniSystem
from cassini.lib.routine import add_routine, parameter_float, parameter_int, parameter_str
from cassini.lib.cassini_utils import ProbeBoard, Gain, ProbeSwitch, Modules, Types, ConnectionStatus, ProberHeight
import cassini.lib.waveform as waveforms
import h5py, time
from datetime import datetime 
import tqdm


# Mathematical functions
import numpy as np

# Plotting 
import matplotlib.pyplot as plt

# Handling of foldersystem
import os

# Dataframes
import pandas as pd

# import Wafermap
import sys

# Import own functions
sys.path.append(r"D:\Scripts\Schnieders\Endurance_measurement_aixACCTtester\functions")
from wafermap_Neurotech1 import wafermap_Neurotech1_1R, wafermap_Neurotech1_1T1R 
from waveforms_cassini import routine_IV_sweep, routine_IV_pulse, add_wf_df
from data_management import main_eval
from algo_management import bool_states, bool_switched, send_msg
from plot_data import make_figures, figure_endurance, get_formatted_datetime

#%% Settings measurements



# Dirty waveform management
df_wf = add_wf_df(None, 'Dummy', ['wf1', 'wf2'], 1, 0)

#####################################################################
## Parameters of measurements
#####################################################################

# Define resistive states
interval_LRS = np.array([0.1, 5])*1e3
interval_HRS = np.array([10, 2e5])*1e3

# Specify sample
sample_layout = "Neurotec1_1R"
sample_material = 'TaO'
sample_name = 'Die_43'
measurement= 'Endurance' + sample_layout

# Save direction
save_dir = os.path.join(r"D:\Data\Schnieders\Endurance" ,
                        sample_layout, sample_material, sample_name, '_'.join(get_formatted_datetime().split('_')[:-3]))

# TODO: Check, if 1R or 1T1R
position, device_names, geometry = wafermap_Neurotech1_1R()

# Parameters switching

# Parameters sweeps
t_break_forming, step_size_forming = 1e-4, 1e-5 # s
sweep_rate_form = 1e2 # V/s

# Parameters sweeps
t_break_sweeps, step_size_sweep = 1e-4, 1e-5 # s
sweep_rate = 1e3 # V/s

V_forming_set, V_forming_reset = 4, -1.7 # V
V_forming_gate = [0, 0]
nr_forming = 1

cc_ps_form, cc_ns_form = 0.15, -3 # mA
V_sweep_set, V_sweep_reset = 1.5, -1.7 # V
V_sweep_gate = [0, 0]
nr_presweeps = 100
nr_sweeps =10

cc_ps, cc_ns = 0.2, -2 # mA
gain_sweep = Gain.LOW

# Parameters pulses
t_break_pulse, t_set_pulse, t_reset_pulse, t_pulse_read, t_sr_pulse = 100e-9, 0.1e-6, 0.8e-6, 0.5e-6, 1e-9 # s
V_pulse_set, V_pulse_reset, V_pulse_read = 1.2, -1.7, 0.2 # V
V_pulse_gate = [0, 0, 0]
cc_pp, cc_np = 0.3, -2 # mA
gain_pulse = Gain.MID


# Number of endurance mesasurements
nr_meas_endurance = [10, 50, 100, 1000, 10000]

bool_LRS, bool_HRS = bool_states(interval_LRS, interval_HRS)
#%% Connect Tester

# We first have to connect to Tester
###############################################################
## Connect Prober
###############################################################
# Instance of Cassini
if 'cassini' not in vars():
    cassini = CassiniSystem()
    try:
        # Contact height and home pos. set
        cassini.connect_prober()
    except:
        pass
    
#%%
    
# Here, we still have a security gap to the real maximum.
MAX_DATAPOINTS_AITESTER = 70e3

def round_base(wf_t, step_size:float):
    '''
    Function for rounding to the stepsize

    Parameters
    ----------
    wf_t: array like 
        time vector that has to be rounded
    step_size: float 
        Base of rounding
    Returns
    -------
    new_wf_t: array like
        Time vector rounded according to  stepsize. 
        If time vector fine, this step also works.
    '''
    delta_wf_t = np.ceil(wf_t/step_size)
    wf_t = np.array([int(t) if int(t) > 0 else 1 for t in delta_wf_t])
    wf_t[0] = 0
    return np.round(np.cumsum(step_size * np.round(wf_t)),9)


def add_wf_df(df_wf, name_wf, list_wf,t_max, nr_rep):
    dict_wf={"name":name_wf,
            'wf':list_wf,
            't_max':t_max,
            'nr_rep': nr_rep
            }
    if type(df_wf) == type(None):
        df_wf = pd.DataFrame([dict_wf])
    else: 
        df_wf = pd.concat([df_wf,pd.DataFrame([dict_wf])], ignore_index = True)
    return df_wf 

# Pulses
def routine_IV_pulse(cassini, 
                     V_set=1, 
                     V_reset=-1,
                     V_read=0.2,
                     cycle=1, 
                     t_set=1e-8,
                     t_reset=1e-8,
                     t_read=1e-8,
                     V_gate=[0,0,0],
                     t_break=1e-7, 
                     n_rep=1,
                     step_size=1e-9,
                     gain=Gain.HIGH, 
                     bool_read=True, 
                     cc_n=-2,
                     cc_p=2,
                     probe_switch=ProbeSwitch.SAMPLE,
                     df_wf=None):
    '''
    

    Parameters
    ----------
    cassini : CassiniSystem
        Cassini object for communication with tester.
    V_set : float
        Set voltage.
    V_reset : float
        Reset voltage.
    V_read : float
        Read voltage.
    cycle : int
        Number cycles.
    t_set : float
        Duration set.
    t_reset : float
        Duration reset.
    t_read : float
        Duration read.
    V_gate : TYPE, optional
        Gate voltages. 
        V_gate[0]: Gatevoltage during first pulse,
        V_gate[1]: Gatevoltage during second pulse,
        The default is [0,0].
    t_break : float, optional
        Waiting time before and between pulses. The default is 1e-7.
    n_rep :int, optional
        Number of pulse sequences in one waveform.
        If n_rep < 1 --> As many cycles as possible.. The default is 1.
    step_size : float, optional
        Time between measurement points. The default is 1e-9.
    gain : cassini.lib.cassini_utils.Gain, optional
        Gain type. The default is Gain.HIGH.
    bool_read: bool, optional
        Add reads to the waveform. The default is True.
    cc_n : float, optional
        Negetive CC. The default is -2.
    cc_p : float, optional
        Positive CC. The default is 2.

    Returns
    -------
    measurement_path : str
        Dir. to raw data.
    measurement_nr : int
        Number of current measurement.

    '''
    cycle = cycle if  cycle < 900 else 900
    cassini.set_cycle(cycle)
    step_size = 2**np.round(np.log2(step_size*1e9))*1e-9
    # We have to assure that step size is following the form: 2**n * 1e-9, n in natural numbers
    # We round the stepsize down to the next value to the form above.
    
    # make sure that only one wf applied

    name_wf = get_formatted_datetime()+'wf_PulseRead_' if bool_read else 'wf_PulseNORead_'
    name_wf = name_wf + f'Vset_{int(V_set*1e3)}mV_Vreset_{int(V_reset*1e3)}mV_tset_{int(t_set*1e9)}ns_treset_{int(t_reset*1e9)}ns_tread_{int(t_read*1e9)}ns_rep_{int(n_rep)}'
    name_wf = name_wf + '_1R' if np.sum(V_gate)==0 else name_wf + '_1T1R'
    if sum(name_wf==df_wf.name)==0:

        if bool_read: 
            #define waveform
            wf_t = np.array([step_size,t_break*3, step_size*8, t_set, 8*step_size,   # set
                t_break, step_size,t_read, step_size,   # read
                t_break, 8*step_size, t_reset, 8*step_size, # reset
                t_break, step_size, t_read, step_size,   # read
                t_break, t_break])

                        
            wf_V = np.array([0, 0,V_set, V_set, 0,               # set
                            0, V_read, V_read, 0,             # read
                            0, V_reset, V_reset, 0,           # reset
                            0, V_read, V_read, 0,             # read
                            0,0])
            if sum(V_gate)>0:
                wf_gate = np.array([0, V_gate[0], V_gate[0], V_gate[0], V_gate[0],
                                    V_gate[2], V_gate[2], V_gate[2], V_gate[2],
                                    V_gate[1], V_gate[1], V_gate[1], V_gate[1],
                                    V_gate[2], V_gate[2], V_gate[2], V_gate[2], 
                                    0])
            else:
                wf_gate=np.array([0,0,0])        
        else: 
            #define waveform
            wf_t = [0, t_break, step_size, t_set, step_size,   # set
                    t_break, step_size, t_reset, step_size, # reset
                    t_break]
            wf_V = np.array([0, V_set, V_set, 0,               # set
                            0, V_reset, V_reset, 0,           # reset
                            0])
            if sum(V_gate)>0:
                wf_gate = np.array([0, V_gate[0], V_gate[0], V_gate[0],
                                    V_gate[1], V_gate[1], V_gate[1], V_gate[1], 
                                    0])

        nr_rep = 1
        # !!! Option to concatenate as many signals as possible. The number of !!! 
        # !!! signals then is given out !!!
        if n_rep>1:
            wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
            wf_t_temp = wf_t_init
            wf_V_temp = wf_V_init
            nr_rep = 1
            for i in range(n_rep-1):
                if max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size: 
                    wf_t_temp = np.append(wf_t_temp[:-1], wf_t_init[2:])
                    wf_V_temp = np.append(wf_V_temp[:-1], wf_V_init[2:])
                    wf_gate = np.append(wf_gate[:-1], wf_gate_init[2:])
                    nr_rep+=1
                wf_t, wf_V = wf_t_temp, wf_V_temp
        # !!! Option to concatenate as many signals as possible. The number of !!! 
        # !!! signals then is given out !!!
        elif n_rep < 1:
            wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
            wf_t_temp = wf_t_init
            wf_V_temp = wf_V_init
            nr_rep = 1
            while max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size:
                wf_t_temp = np.append(wf_t[:-1], wf_t_init[1:]+max(wf_t))
                if max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size: 
                    wf_t = wf_t_temp
                    wf_V = np.append(wf_V[:-1], wf_V_init[1:])
                    wf_gate = np.append(wf_gate[:-1], wf_gate_init[1:])
                    nr_rep+=1
        wf_t = round_base(wf_t, step_size)        
        t_max = max(wf_t)
        if sum(V_gate)==0:
            # Define waveforms
            waveform_iv_sweep = waveforms.Waveform(name_wf, np.array([wf_t, wf_V]),step_size=step_size)
            waveform_ground   = waveforms.Waveform(name_wf+"ground", np.array([wf_t, wf_V*0]),step_size=step_size)
            list_wf = [waveform_iv_sweep, waveform_ground]
            
        else:
            wf_sorce = wf_V*(wf_V>0).astype(int)
            wf_drain = abs(wf_V*(wf_V<0).astype(int))
            waveform_iv_source = waveforms.Waveform(name_wf+"_source", np.array([wf_t, wf_sorce]),step_size=step_size)
            waveform_bulk   = waveforms.Waveform(name_wf+"ground_bulk", np.array([wf_t, wf_V*0]),step_size=step_size)
            waveform_iv_drain = waveforms.Waveform(name_wf+"_drain", np.array([wf_t, wf_drain]),step_size=step_size)
            waveform_gate   = waveforms.Waveform(name_wf+"_gate", np.array([wf_t, wf_V*0]),step_size=step_size)
            list_wf = [waveform_iv_source, waveform_bulk, waveform_iv_drain, waveform_gate]
        
        df_wf = add_wf_df(df_wf, name_wf, list_wf, t_max, nr_rep)

    else: 
        t_max = df_wf.t_max[np.where(name_wf==df_wf.name)[0][0]]
        nr_rep = df_wf.nr_rep[np.where(name_wf==df_wf.name)[0][0]]
        list_wf = df_wf.wf[np.where(name_wf==df_wf.name)[0][0]]
        if sum(V_gate)==0:
            # Define waveforms
            [waveform_iv_sweep, waveform_ground] = list_wf
            
        else:
            [waveform_iv_source, waveform_bulk, waveform_iv_drain, waveform_gate] = list_wf
            
    if sum(V_gate)==0:
        cassini.set_waveform("wedge02", waveform=waveform_iv_sweep)
        cassini.set_waveform("wedge03", waveform=waveform_ground)

    else:
        cassini.set_waveform("wedge01", waveform=waveform_iv_source)
        cassini.set_waveform("wedge02", waveform=waveform_bulk)
        cassini.set_waveform("wedge03", waveform=waveform_iv_drain)
        cassini.set_waveform("wedge04", waveform=waveform_gate)
    
    # set probeboard parameters
    cassini.set_parameter_probeboard(gain=gain, ccn=cc_n, ccp=cc_p,
                                        cc_deactivate=False,probe_switch=probe_switch)

    ## Set ADs
    # First pin
    cassini.set_ad("wedge02", t_max*1.2, termination=True)
    # Second pin
    cassini.set_ad("wedge03", t_max*1.2, termination=True)

    ## Set sampling rate
    cassini.set_recording_samplerate(int(np.round(4/step_size,2)))
    # Measurement
    measurement_path, measurement_nr = cassini.measurement()

    nr_rep = nr_rep*cycle

    return measurement_path, measurement_nr, nr_rep,df_wf

t_break_pulse, t_set_pulse, t_reset_pulse, t_pulse_read, t_sr_pulse = 50e-9, 0.3e-6, 0.5e-6, 0.5e-6, 1e-9 # s
V_pulse_set, V_pulse_reset, V_pulse_read = 1.2, -1.7, 0.2 # V
V_pulse_gate = [0, 0, 0]
nr_meas=2
action = "Switching with read"
max_nr_wf=1
cycle_pulse = int(nr_meas/max_nr_wf) if int(nr_meas/max_nr_wf) > 0 else max_nr_wf
measurement_path, measurement_nr, nr_rep,df_wf = routine_IV_pulse(cassini, 
                                                    V_set= V_pulse_set, 
                                                    V_reset=V_pulse_reset,
                                                    V_read=V_pulse_read,
                                                    cycle= cycle_pulse, 
                                                    t_set=t_set_pulse,
                                                    t_reset=t_reset_pulse,
                                                    t_read=t_pulse_read,
                                                    V_gate=V_pulse_gate,
                                                    t_break=t_break_pulse, 
                                                    n_rep=int(nr_meas) if int(nr_meas)<max_nr_wf else max_nr_wf,
                                                    step_size=t_sr_pulse,
                                                    gain=gain_pulse, 
                                                    bool_read=True, 
                                                    cc_n=cc_np,
                                                    cc_p=cc_pp,
                                                    df_wf=df_wf)
tin, Vin, t, V, I=read_data(measurement_path, measurement_nr,True)

for v in V:
    plt.plot(v)
# %%

#%%
index_site=21
action = "Forming"
# Measurement

def get_formatted_datetime():
    '''
    Return time stamp

    Returns
    -------
    formatted_time : str
        Time stamp with current time.

    '''
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S')
    return formatted_time


# Here, we still have a security gap to the real maximum.
MAX_DATAPOINTS_AITESTER = 70e3

def round_base(wf_t, step_size:float):
    '''
    Function for rounding to the stepsize

    Parameters
    ----------
    wf_t: array like 
        time vector that has to be rounded
    step_size: float 
        Base of rounding
    Returns
    -------
    new_wf_t: array like
        Time vector rounded according to  stepsize. 
        If time vector fine, this step also works.
    '''
    delta_wf_t = np.ceil(wf_t/step_size)
    wf_t = np.array([int(t) if int(t) > 0 else 1 for t in delta_wf_t])
    wf_t[0] = 0
    return np.round(np.cumsum(step_size * np.round(wf_t)),9)


def add_wf_df(df_wf, name_wf, list_wf,t_max, nr_rep):
    dict_wf={"name":name_wf,
            'wf':list_wf,
            't_max':t_max,
            'nr_rep': nr_rep
            }
    if type(df_wf) == type(None):
        df_wf = pd.DataFrame([dict_wf])
    else: 
        df_wf = pd.concat([df_wf,pd.DataFrame([dict_wf])], ignore_index = True)
    return df_wf 



#####################################################################
## Waveforms
#####################################################################

# Sweeps
def routine_IV_sweep(cassini, 
                     V_set=-2, 
                     V_reset=2, 
                     cycle=1, 
                     rate_sweep=1e3,
                     V_gate=[0,0],
                     t_break=1e-7, 
                     n_rep=1,
                     step_size=1e-9,
                     gain=Gain.HIGH, 
                     cc_n=-2,
                     cc_p=2,
                     probe_switch=ProbeSwitch.SAMPLE,
                     df_wf=None
                     ):
    '''
    Function performing sweeps

    Parameters
    ----------
    cassini : CassiniSystem
        Cassini object for communication with tester.
    V_set : float
        Set voltage.
    V_reset : float
        Reset voltage.
    cycle : int
        Number of cycles.
    rate_sweep : float
        Sweep rate.
    V_gate : TYPE, optional
        Gate voltages. 
        V_gate[0]: Gatevoltage during first pulse,
        V_gate[1]: Gatevoltage during second pulse,
        The default is [0,0].
    t_break : float, optional
        Waiting time before and between pulses. The default is 1e-7.
    n_rep :int, optional
        Number of pulse sequences in one waveform.
        If n_rep < 1 --> As many cycles as possible.. The default is 1.
    step_size : float, optional
        Time between measurement points. The default is 1e-9.
    gain : cassini.lib.cassini_utils.Gain, optional
        Gain type. The default is Gain.HIGH.
    cc_n : float, optional
        Negetive CC. The default is -2.
    cc_p : float, optional
        Positive CC. The default is 2.

    Returns
    -------
    measurement_path : str
        Dir. to raw data.
    measurement_nr : int
        Number of current measurement.

    '''
    cycle = cycle if  cycle < 10 else 10
    cassini.set_cycle(cycle)
    step_size = 2**np.round(np.log2(step_size*1e9))*1e-9
    # We have to assure that step size is following the form: 2**n * 1e-9, n in natural numbers
    # We round the stepsize down to the next value to the form above.
    name_wf = get_formatted_datetime()+f'wf_sweep_Vset_{int(V_set*1e3)}mV_Vreset_{int(V_reset*1e3)}mV_rep_{int(n_rep)}_'
    name_wf = name_wf + f'stepsize{int(step_size*1e9)}' 
    name_wf = name_wf + '1R' if np.sum(V_gate)==0 else name_wf + '1T1R'
    if sum(name_wf==df_wf.name)==0:
        # make sure that only one wf applied


        t_set, t_reset = abs(V_set/rate_sweep), abs(V_reset/rate_sweep)
        #define waveform
        wf_V = np.array([0, 0, V_set, 0, 0,
                        V_reset,0,0])
        
        wf_t = np.array([0, step_size*50, t_set , t_set, step_size*10,
                t_reset, t_reset, t_break])

        # If we define a gate voltage apart from 0V, we also define a waveform for the gate
        if sum(V_gate)>0:
            wf_gate = np.array([0, V_gate[0], V_gate[0], V_gate[0], V_gate[1],
                        V_gate[1],V_gate[1],0])
        else:
            wf_gate =  np.array([0,0,0])
        nr_rep = 1

        # Concatenate the same waveform together
        # If nr. of repetitions fixed, then concatenate this many. (If waveform not to long.)
        if n_rep>1:
            wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
            wf_t_temp = wf_t_init
            wf_V_temp = wf_V_init
            nr_rep = 1
            for i in range(n_rep-1):
                if max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size: 
                    wf_t_temp = np.append(wf_t_temp[:-1], wf_t_init[1:])
                    wf_V_temp = np.append(wf_V_temp[:-1], wf_V_init[1:])
                    wf_gate = np.append(wf_gate[:-1], wf_gate_init[1:])
                    nr_rep+=1
                wf_t, wf_V = wf_t_temp, wf_V_temp
        # !!! Option to concatenate as many signals as possible. The number of !!! 
        # !!! signals then is given out !!!
        elif n_rep < 1:
            wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
            wf_t_temp = wf_t_init
            wf_V_temp = wf_V_init
            nr_rep = 1
            while max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size:
                wf_t_temp = np.append(wf_t[:-1], wf_t_init[1:]+max(wf_t))
                if max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size: 
                    wf_t = wf_t_temp
                    wf_V = np.append(wf_V[:-1], wf_V_init[1:])
                    wf_gate = np.append(wf_gate[:-1], wf_gate_init[1:])
                    nr_rep+=1
        wf_t = round_base(wf_t, step_size)
        t_max = max(wf_t)
        # We have to load signals into different channels according to structure.
        if sum(V_gate)==0:
            # Define waveforms
            waveform_iv_sweep = waveforms.Waveform(name_wf, np.array([wf_t, wf_V]),step_size=step_size)
            waveform_ground   = waveforms.Waveform(name_wf+"ground", np.array([wf_t, wf_V*0]),step_size=step_size)
            list_wf = [waveform_iv_sweep, waveform_ground]
            
        else:
            wf_sorce = abs(wf_V*(wf_V>0).astype(float))
            wf_drain = abs(wf_V*(wf_V<0).astype(float))
            waveform_iv_source = waveforms.Waveform(name_wf+"_source", np.array([wf_t, wf_sorce]),step_size=step_size)
            waveform_bulk   = waveforms.Waveform(name_wf+"ground_bulk", np.array([wf_t, wf_V*0]),step_size=step_size)
            waveform_iv_drain = waveforms.Waveform(name_wf+"_drain", np.array([wf_t, wf_drain]),step_size=step_size)
            waveform_gate   = waveforms.Waveform(name_wf+"_gate", np.array([wf_t, wf_gate]),step_size=step_size)
            list_wf = [waveform_iv_source, waveform_bulk, waveform_iv_drain, waveform_gate]
        df_wf = add_wf_df(df_wf, name_wf, list_wf, max(wf_t), nr_rep)
    else: 
        t_max = df_wf.t_max[np.where(name_wf==df_wf.name)[0][0]]
        nr_rep = df_wf.nr_rep[np.where(name_wf==df_wf.name)[0][0]]
        list_wf = df_wf.wf[np.where(name_wf==df_wf.name)[0][0]]
        if sum(V_gate)==0:
            # Define waveforms
            [waveform_iv_sweep, waveform_ground] = list_wf
            
        else:
            [waveform_iv_source, waveform_bulk, waveform_iv_drain, waveform_gate] = list_wf
        
    if sum(V_gate)==0:
        cassini.set_waveform("wedge02", waveform=waveform_iv_sweep)
        cassini.set_waveform("wedge03", waveform=waveform_ground)

    else:
        cassini.set_waveform("wedge01", waveform=waveform_iv_source)
        cassini.set_waveform("wedge02", waveform=waveform_bulk)
        cassini.set_waveform("wedge03", waveform=waveform_iv_drain)
        cassini.set_waveform("wedge04", waveform=waveform_gate)

    # set probeboard parameters
    cassini.set_parameter_probeboard(gain=gain, ccn=cc_n, ccp=cc_p,
                                     cc_deactivate=False,probe_switch=probe_switch)

    ## Set ADs
    # First pin
    cassini.set_ad("wedge02", t_max*1.2, termination=True)
    # Second pin
    cassini.set_ad("wedge03", t_max*1.2, termination=True)

    ## Set sampling rate
    cassini.set_recording_samplerate(int(np.round(1/step_size,2)))
    # Measurement
    time.sleep(0.5)
    measurement_path, measurement_nr = cassini.measurement()

    nr_rep = n_rep*cycle

    return measurement_path, measurement_nr, n_rep, df_wf

V_forming_gate_curr, bool_meas_1T1R = choose_gate_V(V_forming_gate, geometry[index_site])

measurement_path, measurement_nr, nr_rep, df_wf = routine_IV_sweep(cassini, 
                        V_forming_set, 
                        V_forming_reset,
                        cycle=1,
                        rate_sweep= sweep_rate_form,
                        V_gate=V_forming_gate_curr,
                        t_break=t_break_forming, 
                        n_rep=1,
                        step_size=step_size_forming,
                        gain=gain_sweep, 
                        cc_n=cc_ns_form, 
                        cc_p=cc_ps_form,
                         df_wf=df_wf)


tin, Vin, tread, Vdrain, Vsrc, Vgate, Iread  = read_data(measurement_path, measurement_nr,bool_not1T1R='1T1R' not  in dir_device)
for v in Iread:
    plt.plot(v)
plt.show()

# %%
