# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:52:59 2024

@author: schnieders
"""

#####################################################################
## Import functions
#####################################################################

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
sys.path.append("DIR with scripts")
from wafermap_Neurotech1 import wafermap_Neurotech1_1R, wafermap_Neurotech1_1T1R 
from waveforms_cassini import routine_IV_sweep, routine_IV_pulse
from data_management import main_eval
from algo_management import bool_states, bool_switched

#####################################################################
## Parameters of measurements
#####################################################################

# Define resistive states
interval_LRS = np.array([2, 5])*1e3
interval_HRS = np.array([20, 100])*1e3

# Specify sample
sample_layout = "Neurotec_cat" # Mohit
sample_material = 'cute_cat'
sample_name = 'cat'
measurement= 'Endurance' + sample_layout

# Save direction
save_dir = os.path.join(r"\\iff1690.iff.kfa-juelich.de\data2\Data\Schnieders\Endurance" ,
                        sample_layout, sample_material, sample_name)
position, device_names, geometry = wafermap_Neurotech1_1R

# Parameters switching

# Parameters sweeps
t_break_sweeps, step_size_sweep = 1e-4, 1e-5 # s
sweep_rate = 1e3 # V/s

V_forming_set, V_forming_reset = 3, -2 # V
V_forming_gate = [0, 0]
nr_forming = 1

V_sweep_set, V_sweep_reset = 1.5, -1.5 # V
V_sweep_gate = [0, 0]
nr_presweeps = 100

cc_p, cc_n = 0.2, -2 # mA
gain_sweep = Gain.MID

# Parameters pulses
t_break_pulse, t_set_pulse, t_reset_pulse, t_pulse_read = 50e-9, 1e-6, 10e-6, 1e-6 # s
V_pulse_set, V_pulse_reset, V_pulse_read = 1.5, -1.5, 0.2 # V
V_pulse_gate = [0, 0, 0]
cc_p, cc_n = 0.2, -2 # mA
gain_pulse = Gain.MID

# Number of endurance mesasurements
nr_meas_endurance = [0, 1, 5, 10, 50, 100, 1000]

bool_LRS, bool_HRS = bool_states(interval_LRS, interval_HRS)

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
    
# Metadata
cassini.set_meta(operator="cute.cat", wafer_name=sample_layout+ '_'+ sample_material + '_'+ sample_name + '_'+ measurement, orientation=0)

# We first have to connect to Tester
###############################################################
## Connect Prober
###############################################################
# Instance of Cassini

for index_site, device_name in enumerate(device_names): 
    
    # Save dir of device
    dir_device = os.path.join(save_dir, device_name)
    
    # Make sure, the dir. for the device exists. 
    os.makedirs(dir_device, exist_ok=True)
    
    ###############################################################
    ## Forming
    ###############################################################

    action = "Forming"
    # Measurement
    measurement_path, measurement_nr, nr_rep = routine_IV_sweep(cassini, 
                         V_forming_set, 
                         V_forming_reset,
                         nr_forming,  # Nr. cycles
                         sweep_rate,
                         V_gate=V_forming_gate,
                         t_break=t_break_sweeps, 
                         n_rep=1,
                         step_size=step_size_sweep,
                         gain=gain_sweep, 
                         cc_n=cc_n, 
                         cc_p=cc_p)
    # Evaluate measurement 
    R_states = main_eval(dir_device, 
                  measurement_path, 
                  measurement_nr, 
                  action, 
                  device_name, 
                  bool_sweep=True)
    # Verify if forming successful
    bool_formed = bool_switched(R_states[1], R_states[3], bool_LRS, bool_HRS)
    
    # If the device is not formed, we go on.
    if not bool_formed:
        continue
    else:
        n_switch=1
    
    ###############################################################
    ## Sweeps
    ###############################################################
    
    action = "Sweep"
    # Measurement
    measurement_path, measurement_nr, nr_rep = routine_IV_sweep(cassini, 
                         V_sweep_set, 
                         V_sweep_reset,
                         nr_presweeps,  # Nr. cycles
                         sweep_rate,
                         V_gate=V_sweep_gate,
                         t_break=t_break_sweeps, 
                         n_rep=1,
                         step_size=step_size_sweep,
                         gain=gain_sweep, 
                         cc_n=cc_n, 
                         cc_p=cc_p)
    
    # Evaluate measurement 
    R_states = main_eval(dir_device, 
                  measurement_path, 
                  measurement_nr, 
                  action, 
                  device_name, 
                  bool_sweep=True)
    
    
    # Verify if forming successful
    nr_sweep_switched = sum([bool_switched(R_states[i*4+1], R_states[i*4+3], bool_LRS, bool_HRS) for i in range(int(len(R_states)/4))])
    if nr_sweep_switched<=nr_presweeps*0.8:
        continue
    else:
        n_switch+=nr_sweep_switched
        
    ###############################################################
    ## Pulses/ Start endurance
    ###############################################################
    bool_device_working, id_nr = True, 0
    while bool_device_working:
        nr_meas = nr_meas_endurance[id_nr]
        n_dummy=0
        if nr_meas >10: 
            measurement_path, measurement_nr, nr_rep = routine_IV_pulse(cassini, 
                                                                V_pulse_set, 
                                                                V_pulse_reset,
                                                                V_pulse_read,
                                                                nr_meas, 
                                                                t_set_pulse,
                                                                t_reset_pulse,
                                                                t_pulse_read,
                                                                V_gate=V_pulse_gate,
                                                                t_break=t_break_pulse, 
                                                                n_rep=-1,
                                                                step_size=t_break_pulse,
                                                                gain=gain_pulse, 
                                                                bool_read=False, 
                                                                cc_n=cc_n,
                                                                cc_p=cc_p)
            n_dummy+=nr_rep
        measurement_path, measurement_nr, nr_rep = routine_IV_pulse(cassini, 
                                                            V_pulse_set, 
                                                            V_pulse_reset,
                                                            V_pulse_read,
                                                            nr_meas, 
                                                            t_set_pulse,
                                                            t_reset_pulse,
                                                            t_pulse_read,
                                                            V_gate=V_pulse_gate,
                                                            t_break=t_break_pulse, 
                                                            n_rep=nr_meas if nr_meas<50 else 10,
                                                            step_size=t_break_pulse,
                                                            gain=gain_pulse, 
                                                            bool_read=True, 
                                                            cc_n=cc_n,
                                                            cc_p=cc_p)
        # Evaluate measurement 
        R_states = main_eval(dir_device, 
                      measurement_path, 
                      measurement_nr, 
                      action, 
                      device_name, 
                      bool_sweep=False)
        
        # Verify if forming successful
        nr_pulse_switched = sum([bool_switched(R_states[i*2], R_states[i*2+1], bool_LRS, bool_HRS) for i in range(int(len(R_states)/4))])
        if nr_pulse_switched<=nr_rep*0.8:
            bool_device_working=False
        else:
            n_switch+=nr_pulse_switched+nr_rep
            