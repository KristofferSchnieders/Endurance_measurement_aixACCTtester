# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:52:59 2024

@author: schnieders
"""


## Communication with setup
from cassini.system import CassiniSystem
from cassini.lib.routine import add_routine, parameter_float, parameter_int, parameter_str
from cassini.lib.cassini_utils import ProbeBoard, Gain, ProbeSwitch, Modules, Types, ConnectionStatus, ProberHeight
import cassini.lib.waveform as waveforms

# Mathematical functions
import numpy as np

import sys 

import copy

sys.path.append("DIR with scripts")
from data_management import *
from algo_management import bool_states

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

    return np.cumsum(step_size * np.round(wf_t))


#####################################################################
## Waveforms
#####################################################################

# Sweeps
def routine_IV_sweep(cassini, 
                     V_set:float, 
                     V_reset:float, 
                     cycle=1, 
                     rate_sweep=1e3,
                     V_gate=[0,0],
                     t_break=1e-7, 
                     n_rep=1,
                     step_size=1e-9,
                     gain=Gain.HIGH, 
                     cc_n=-2,
                     cc_p=2,
                     probe_switch=ProbeSwitch.SAMPLE):
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

    # We have to assure that step size is following the form: 2**n * 1e-9, n in natural numbers
    # We round the stepsize down to the next value to the form above.
    step_size = 2**np.round(np.log2(step_size*1e9))*1e-9

    # make sure that only one wf applied
    cassini.set_cycle(cycle)

    t_set, t_reset = abs(V_set/rate_sweep), abs(V_reset/rate_sweep)
    #define waveform
    wf_V = np.array([0, 0, V_set, 0, 0,
                      V_reset,0,0])
    
    wf_t = np.array([0, t_break, t_set , t_set, step_size*10,
               t_reset, t_reset, t_break])
    wf_t = round_base(wf_t, step_size)

    # If we define a gate voltage apart from 0V, we also define a waveform for the gate
    if sum(V_gate)>0:
        wf_gate = np.array([0, V_gate[0], V_gate[0], V_gate[0], V_gate[1],
                      V_gate[1],V_gate[1],0])
    else:
        wf_gate =  np.array([0,0,0])
    
    # Concatenate the same waveform together
    # If nr. of repetitions fixed, then concatenate this many. (If waveform not to long.)
    if n_rep>1:
        wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
        wf_t_temp = wf_t_init
        wf_V_temp = wf_V_init
        nr_rep = 1
        for i in range(n_rep-1):
            if max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size: 
                wf_t_temp = np.append(wf_t_temp[:-1], wf_t_init[1:]+max(wf_t_temp))
                wf_V_temp = np.append(wf_V_temp[:-1], wf_V_init[1:])
                wf_gate = np.append(wf_gate[:-1], wf_gate_init[1:])
                nr_rep+=1
    # !!! Option to concatenate as many signals as possible. The number of !!! 
    # !!! signals then is given out !!!
    elif n_rep < 1:
        wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
        wf_t_temp = wf_t_init
        wf_V_temp = wf_V_init
        nr_rep = 1
        while max(wf_t) < MAX_DATAPOINTS_AITESTER*step_size:
            wf_t_temp = np.append(wf_t[:-1], wf_t_init[1:]+max(wf_t))
            if wf_t_temp < MAX_DATAPOINTS_AITESTER*step_size: 
                wf_t = wf_t_temp
                wf_V = np.append(wf_V[:-1], wf_V_init[1:])
                wf_gate = np.append(wf_gate[:-1], wf_gate_init[1:])
                nr_rep+=1
    # set probeboard parameters
    cassini.set_parameter_probeboard(gain=gain, ccn=cc_n, ccp=cc_p,
                                     cc_deactivate=False,probe_switch=probe_switch)
    
    # Define waveforms
    waveform_iv_sweep = waveforms.Waveform("sweep", np.array([wf_t, wf_V]),step_size=step_size)
    waveform_ground   = waveforms.Waveform("ground", np.array([wf_t, wf_V*0]),step_size=step_size)
    
    # We have to load signals into different channels according to structure.
    if sum(V_gate)==0:
    
        #Set DAs
        cassini.set_waveform("wedge02", waveform=waveform_iv_sweep)
        cassini.set_waveform("wedge03", waveform=waveform_ground)

    else:
        waveform_gate     = waveforms.Waveform("gate", np.array([wf_t, wf_gate]),step_size=step_size)

        cassini.set_waveform("wedge01", waveform=waveform_ground)
        cassini.set_waveform("wedge02", waveform=waveform_gate)
        cassini.set_waveform("wedge03", waveform=waveform_iv_sweep)
        cassini.set_waveform("wedge04", waveform=waveform_ground)


    ## Set ADs
    # First pin
    cassini.set_ad("wedge02", max(wf_t)*1.2, termination=True)
    # Second pin
    cassini.set_ad("wedge03", max(wf_t)*1.2, termination=True)

    ## Set sampling rate
    print(int(step_size))
    cassini.set_recording_samplerate(int(1/step_size))
    # Measurement
    measurement_path, measurement_nr = cassini.measurement()

    return measurement_path, measurement_nr, n_rep

# Pulses
def routine_IV_pulse(cassini, 
                     V_set: float, 
                     V_reset: float,
                     V_read: float,
                     cycle: int, 
                     t_set: float,
                     t_reset: float,
                     t_read: float,
                     V_gate=[0,0,0],
                     t_break=1e-7, 
                     n_rep=1,
                     step_size=1e-9,
                     gain=Gain.HIGH, 
                     bool_read=True, 
                     cc_n=-2,
                     cc_p=2):
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

    # We have to assure that step size is following the form: 2**n * 1e-9, n in natural numbers
    # We round the stepsize down to the next value to the form above.
    step_size = 2**np.round(np.log2(step_size*1e9))*1e-9

    # make sure that only one wf applied
    cassini.set_cycle(cycle)


    if bool_read: 
        #define waveform
        wf_t = np.array([0, t_break, step_size, t_set, step_size,   # set
                   t_break, step_size, t_read, step_size,   # read
                   t_break, step_size, t_reset, step_size, # reset
                   t_break, step_size, t_read, step_size,   # read
                   t_break])
        wf_t = np.round(np.cumsum(wf_t),9)
        wf_V = np.array([0, V_set, V_set, 0,               # set
                         0, V_read, V_read, 0,             # read
                         0, V_reset, V_reset, 0,           # reset
                         0, V_read, V_read, 0,             # read
                         0])
        if sum(V_gate)>0:
            wf_gate = np.array([0, V_gate[0], V_gate[0], V_gate[0],
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
        wf_t = round_base(wf_t, step_size)
        wf_V = np.array([0, V_set, V_set, 0,               # set
                         0, V_reset, V_reset, 0,           # reset
                         0])
        if sum(V_gate)>0:
            wf_gate = np.array([0, V_gate[0], V_gate[0], V_gate[0],
                                V_gate[1], V_gate[1], V_gate[1], V_gate[1], 
                                0])
    wf_t = round_base(wf_t, step_size)
    
    # !!! Option to concatenate as many signals as possible. The number of !!! 
    # !!! signals then is given out !!!
    if n_rep>1:
        wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
        wf_t_temp = wf_t_init
        wf_V_temp = wf_V_init
        nr_rep = 1
        for i in range(n_rep-1):
            if max(wf_t_temp) < MAX_DATAPOINTS_AITESTER*step_size: 
                wf_t_temp = np.append(wf_t_temp[:-1], wf_t_init[1:]+max(wf_t_temp))
                wf_V_temp = np.append(wf_V_temp[:-1], wf_V_init[1:])
                wf_gate = np.append(wf_gate[:-1], wf_gate_init[1:])
                nr_rep+=1
    # !!! Option to concatenate as many signals as possible. The number of !!! 
    # !!! signals then is given out !!!
    elif n_rep < 1:
        wf_t_init, wf_V_init, wf_gate_init = copy.deepcopy(wf_t), copy.deepcopy(wf_V), copy.deepcopy(wf_gate)
        wf_t_temp = wf_t_init
        wf_V_temp = wf_V_init
        nr_rep = 1
        while max(wf_t) < MAX_DATAPOINTS_AITESTER*step_size:
            wf_t_temp = np.append(wf_t[:-1], wf_t_init[1:]+max(wf_t))
            if wf_t_temp < MAX_DATAPOINTS_AITESTER*step_size: 
                wf_t = wf_t_temp
                wf_V = np.append(wf_V[:-1], wf_V_init[1:])
                wf_gate = np.append(wf_gate[:-1], wf_gate_init[1:])
                nr_rep+=1
    
    # set probeboard parameters
    cassini.set_parameter_probeboard(gain=gain, ccn=cc_n, ccp=cc_p,
                                     cc_deactivate=False)
    
    # Define waveforms
    waveform_iv_sweep = waveforms.Waveform("sweep", np.array([wf_t, wf_V]),step_size=step_size)
    waveform_ground   = waveforms.Waveform("ground", np.array([wf_t, wf_V*0]),step_size=step_size)
    
    
    # We have to load signals into different channels according to structure.
    if sum(V_gate)>0:
        waveform_gate     = waveforms.Waveform("gate", np.array([wf_t, wf_gate]),step_size=step_size)


    #Set DAs
    cassini.set_waveform("wedge02", waveform=waveform_iv_sweep)
    cassini.set_waveform("wedge03", waveform=waveform_ground)

    if sum(V_gate)>0:
        cassini.set_waveform("wedge01", waveform=waveform_ground)
        cassini.set_waveform("wedge02", waveform=waveform_gate)
        cassini.set_waveform("wedge03", waveform=waveform_iv_sweep)
        cassini.set_waveform("wedge04", waveform=waveform_ground)

    # set probeboard parameters
    cassini.set_parameter_probeboard(gain=gain, ccn=cc_n, ccp=cc_p,
                                     cc_deactivate=False)

    ## Set ADs
    # First pin
    cassini.set_ad("wedge02", max(wf_t)*1.2, termination=True)
    # Second pin
    cassini.set_ad("wedge03", max(wf_t)*1.2, termination=True)

    ## Set sampling rate
    cassini.set_recording_samplerate(int(1/step_size))
    # Measurement
    measurement_path, measurement_nr = cassini.measurement()

    return measurement_path, measurement_nr, nr_rep

