# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:52:59 2024

@author: schnieders
"""
#%% Imports

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
t_break_pulse, t_set_pulse, t_reset_pulse, t_pulse_read, t_sr_pulse = 120e-9, 0.12e-6, 0.8e-6, 0.52e-6, 4e-9 # s
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
    
    
# Metadata
cassini.set_meta(operator="k.schnieders", wafer_name=sample_layout+ '_'+ sample_material + '_'+ sample_name + '_'+ measurement, orientation=0)

# We first have to connect to Tester
###############################################################
## Connect Prober
###############################################################
# Instance of Cassini
#%% Measurements
cassini.prober.move_height_level(ProberHeight.CONTACT)
#TODO: Decide, which devices should be chosen.
id_device_offset, id_max_device, id_step_device = 8, 378, 20


# This is a trick to ensure that a Telegram message saying that there was an error is send to me. 
# !!!!!!! ?Is there a way to get the complete errormessage as string and process this? !!!!!!!
#try:
for id_device, device_name in enumerate(device_names[id_device_offset:id_max_device:id_step_device]): 
    if id_device<5:
        cc_ps_form, cc_ns_form = 0.2, -3 # mA
    else:
        cc_ps_form, cc_ns_form = 0.1, -3 # mA
    print(id_device)
    cassini.prober.move_height_level(ProberHeight.CONTACT)
    index_site = id_device*id_step_device+id_device_offset
    cassini.prober.goto(0,0,index_site)
    # Save dir of device
    dir_device = os.path.join(save_dir, device_name[0], device_name[1])
    
    # Make sure, the dir. for the device exists. 
    os.makedirs(dir_device, exist_ok=True)
    
    ###############################################################
    ## Forming
    ###############################################################

    action = "Forming"
    # Measurement
    
    measurement_path, measurement_nr, nr_rep, df_wf = routine_IV_sweep(cassini, 
                            V_forming_set, 
                            V_forming_reset,
                            cycle=1,
                            rate_sweep= sweep_rate_form,
                            V_gate=V_forming_gate,
                            t_break=t_break_forming, 
                            n_rep=1,
                            step_size=step_size_forming,
                            gain=gain_sweep, 
                            cc_n=cc_ns_form, 
                            cc_p=cc_ps_form,
                            df_wf=df_wf)

    # Evaluate measurement 
    R_states, df_endurance, states = main_eval(dir_device, 
                    measurement_path, 
                    measurement_nr, 
                    action, 
                    device_name, 
                    bool_sweep=True,
                    df_endurance=None)
    
    # Verify if forming successful
    nr_pulse_switched = sum(bool_switched(R_states, states, bool_LRS, bool_HRS)[0])
    
    # If the device is not formed, we go on.
    if not bool_LRS(R_states[0]):
        continue
    else:
        n_switch=nr_pulse_switched
    
    ###############################################################
    ## Sweeps
    ###############################################################
    
    action = "Sweep"
    # Measurement
    cycle_sweep = int(nr_presweeps/10) if int(nr_presweeps/10) > 0 else 1
    n_presweeps =  int(np.round(nr_presweeps/5,0))
    counter_presweep=0
    while counter_presweep < nr_presweeps:
        measurement_path, measurement_nr, nr_rep, df_wf = routine_IV_sweep(cassini, 
                                V_sweep_set, 
                                V_sweep_reset,
                                cycle=cycle_sweep if cycle_sweep<2 else 1,  # Nr. cycles
                                rate_sweep=sweep_rate,
                                V_gate=V_sweep_gate,
                                t_break=t_break_sweeps, 
                                n_rep=n_presweeps,
                                step_size=step_size_sweep,
                                gain=gain_sweep, 
                                cc_n=cc_ns, 
                                cc_p=cc_ps,
                                df_wf=df_wf)
        
        # Evaluate measurement 
        R_states, df_endurance, states = main_eval(dir_device, 
                        measurement_path, 
                        measurement_nr, 
                        action, 
                        device_name, 
                        bool_sweep=True,
                        df_endurance=df_endurance)
        counter_presweep += nr_rep
    
    # Verify if forming successful
    nr_sweep_switched = sum(bool_switched(R_states, states, bool_LRS, bool_HRS)[0])
    if nr_sweep_switched<=nr_rep*0.8:
        continue
    else:
        n_switch+=nr_sweep_switched
        
    ###############################################################
    ## Pulses/ Start endurance
    ###############################################################
    bool_device_working, id_nr, counter_sweep = True, 0, 100
    while bool_device_working:
        
        # Unsafe way to limit the number of pulses to the maximal number in the list
        try:
            nr_meas = nr_meas_endurance[id_nr]
        except:
            nr_meas = nr_meas_endurance[-1]
        n_dummy=0
        # Currently not working
        if nr_meas >10 and False: 
            action =  "Pulse"
            measurement_path, measurement_nr, nr_rep, df_wf = routine_IV_pulse(cassini, 
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
                                                                cc_n=cc_np,
                                                                cc_p=cc_pp,
                                                                df_wf=df_wf)
            n_dummy+=nr_rep
        action = "Switching with read"
        
        cycle_pulse = int(nr_meas/3) if int(nr_meas/3) > 0 else 1
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
                                                            n_rep=int(nr_meas) if int(nr_meas)<3 else 3,
                                                            step_size=t_sr_pulse,
                                                            gain=gain_pulse, 
                                                            bool_read=True, 
                                                            cc_n=cc_np,
                                                            cc_p=cc_pp,
                                                            df_wf=df_wf)
        # Evaluate measurement 
        R_states, df_endurance, states = main_eval(dir_device, 
                        measurement_path, 
                        measurement_nr, 
                        action, 
                        device_name, 
                        bool_sweep=False,
                        df_endurance=df_endurance)
        
        # Verify if forming successful
        nr_pulse_switched = sum(bool_switched(R_states, states, bool_LRS, bool_HRS)[0])
        if nr_pulse_switched<nr_rep*0.8:
            bool_device_working=False
            
        else:
            n_switch+=nr_pulse_switched+(n_dummy)*(nr_pulse_switched/nr_rep)
            id_nr+=1
        
        
        if n_switch>counter_sweep:
            ###############################################################
            ## Sweeps
            ###############################################################
            
            action = "Sweep"
            # Measurement
            measurement_path, measurement_nr, nr_rep, df_wf = routine_IV_sweep(cassini, 
                                    V_sweep_set, 
                                    V_sweep_reset,
                                    cycle=nr_sweeps,  # Nr. cycles
                                    rate_sweep=sweep_rate,
                                    V_gate=V_sweep_gate,
                                    t_break=t_break_sweeps, 
                                    n_rep=10,
                                    step_size=step_size_sweep,
                                    gain=gain_sweep, 
                                    cc_n=cc_ns, 
                                    cc_p=cc_ps,
                                    df_wf=df_wf)
            
            # Evaluate measurement 
            R_states, df_endurance, states = main_eval(dir_device, 
                            measurement_path, 
                            measurement_nr, 
                            action, 
                            device_name, 
                            bool_sweep=True,
                            df_endurance=df_endurance)
            
            
            # Verify if forming successful
            nr_sweep_switched = sum(bool_switched(R_states, states, bool_LRS, bool_HRS)[0])

            n_switch+=nr_sweep_switched
            counter_sweep *= 2
    
    figure_endurance(df_endurance, states, "Block " + device_name[0] + " device " + device_name[1], dir_device)
"""
except: 
    send_msg("There was an error. Please check what happend.")
    print("Please save the complete error message.")
    We_want_to_have_an_error_here     
"""
cassini.prober.goto(0,0,0)
cassini.prober.move_height_level(ProberHeight.SEPARATION)
send_msg("Measurement ended without error.")
# %%