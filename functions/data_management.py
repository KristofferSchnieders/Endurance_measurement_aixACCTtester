# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:07:56 2024

@author: schnieders
"""

import numpy as np 
import h5py
from datetime import datetime
import pandas as bearcats
import os
import sys 

sys.path.append(r"D:\Scripts\Schnieders\Endurance_measurement_aixACCTtester\functions")
from plot_data import *


# Read the measured data
def read_data(measurement_path: str, measurement_nr: int, bool_not1T1R=True):
    '''
    Read data from HDF5

    Parameters
    ----------
    measurement_path : str
        Path to HDF5.
    measurement_nr : int
        Number of measurement.

    Returns
    -------
    tin : array like
        Time vector defined / s.
    Vin : array like
        Voltage vector defined / V.
    tread : array like
        Measured time / s.
    Vread : array like
        Measured voltage / V.
    Iread : array like
        Measure current / A.

    '''
    
    # Makes use also possible if used from IFF network computer
    if not os.path.isfile(measurement_path):
        measurement_path = os.path.join(r"\\iff1690.iff.kfa-juelich.de\data2",os.path.join(*measurement_path.split(r"/")[1:]))
    if not os.path.isfile(measurement_path):
        raise FileNotFoundError("You are not inside the IFF system. You cannot directly access the data or have to enter the access code for the computer.")
        

    if bool_not1T1R:
        with h5py.File(measurement_path ,"r") as f:
            tin =np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_DA_wedge02"][0])
            Vin =np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_DA_wedge02"][1])
            tread = np.array([t[0] for t in np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_AD_wedge02"])])
            Vread = np.array([V[1] for V in np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_AD_wedge02"])])
            Iread = np.array([I[1] for I in np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_AD_wedge03"])])
    else:
        V_in_src =np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_DA_wedge01"])
        V_in_drain =np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_DA_wedge03"])
        V_in_gate =np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_DA_wedge04"])
        
        Iread =-np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_AD_wedge03"])[0]
        
        Vgate = np.interp(Iread[0],V_in_gate[0],V_in_gate[1])
        Vsrc = np.interp(Iread[0],V_in_src[0],V_in_src[1])
        Vdrain = np.interp(Iread[0],V_in_drain[0],V_in_drain[1])
        Vread = Vsrc - Vdrain

    if bool_not1T1R:
        return tin, Vin, tread, Vread, Iread
    else: 
        return tin, Vin, tread, Vread, Iread, Vgate

def smooth_mean(x, N=10):
    '''
    We want to smooth the data. Due to floating point errors, we have to cannot do this with cumsum
    Parameters:
    ----------
        x: array like
            Data  to be smoothed 
        N: int
            Number of datapoints to average over
    Returns:
   ----------
        smoothedList: array like
            Smoothed data
    '''
    smoothList = list()
    lenx = len(x)/1000
    if lenx > 1000:
        for id_smooth in range(1000): 
            cumsum = np.cumsum(x[int(id_smooth*lenx): np.min([int((1+id_smooth)*lenx), len(x)-1])])
            if id_smooth == 0: 
                smoothList = smoothList + list((cumsum[N::N]-cumsum[:-N:N]) / N)
            else: 
                smoothList = smoothList + list((cumsum[N::N]-cumsum[:-N:N]) / N)
            
    else:
        cumsum = np.cumsum(x)
        smoothList = (cumsum[N::N]-cumsum[:-N:N]) / N

    return np.array(smoothList)

def running_median(data, n_runmed=10):
    '''
    Can take quite some time for longer array.

    Parameters
    ----------
    data : array like
        List with data.
    n_runmed : int, optional
        Number of data points to take median over. The default is 10.

    Returns
    -------
    array like
        Filtered data.

    '''
    return np.array([np.median(data[index:(index+n_runmed)]) for index in
                     range(len(data))])

def get_states(Vin, Vread=0.2):
    states= list()
    for i, v_in in enumerate(Vin[:-1]):
        if v_in==0 and Vin[i+1]==Vread:
            Vpulse = Vin[i-4:i][np.argmax(abs(Vin[i-4:i]))]
            states.append('LRS' if Vpulse>0 else 'HRS')
    return states
# Main function for filtering of data
def filter_data(I, V, t, Nmean=5):
    '''
    Filter data

    Parameters
    ----------
    I : array like
        Measured current / A 
    V : array like
        Measured voltage / V
    t : array like
        Measured time / s
    Nmean : int, optional
        Number of datapoints to average over. The default is 5.

    Returns
    -------
    I_filt : array like
        Filtered current / A 
    V_filt : array like
        Filtered voltage / V 
    t_filt : array like
        filtered time / s 

    '''
    N_smooth = 2
    I_filt, V_filt, t_filt = list(), list(), list()
    for i, dummy_I in enumerate(I):
        indexV_0 = np.where(abs(V[i])<0.05)[0]
        I_offset = np.mean(I[i][indexV_0][-50:]) if abs(np.mean(I[i][indexV_0][-10:]))<5e-5 else -4e-5 #np.mean(I[i][:10])
        I_filt_dummy=smooth_mean(I[i]-I_offset,N=N_smooth)
        I_filt.append(running_median(I_filt_dummy[Nmean:],n_runmed=4))
        
        V_offset = np.mean(V[i][indexV_0][-10:])
        V_filt_dummy=smooth_mean(V[i]-V_offset,N=N_smooth)

        V_filt.append(running_median(V_filt_dummy[Nmean:],n_runmed=4))
        t_filt.append(np.linspace(0,max(t[i]), len(V_filt[-1])))

    return I_filt, V_filt, t_filt

#####################################################################
## Define Datetime
#####################################################################

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

#####################################################################
## Save raw data
#####################################################################
def load_raw_data_and_store(measurement_path, 
                            measurement_nr,
                            action,
                            device_name,
                            dir_data_save=r"\\iff1690.iff.kfa-juelich.de\data2\Data\Schnieders\Endurance\cute_cat"):
    '''
    Function for storing some data about the measurement in a seperate file. 
    Here it is debatable, if raw data should be stored. 

    Parameters
    ----------
    measurement_path : str
        Path to HDF5.
    measurement_nr : int
        Number of measurement.
    action : str
        Identifier for kind of measurement. 
    id_site : inta
        Identifier of current device. (Only reasonable if combined with wafermap.)
    dir_data_save : str, optional
        Dir. in which we want to store the data. The default is r"\\iff1690.iff.kfa-juelich.de\data2\Data\Schnieders\Endurance\cute_cat".

    Returns
    -------
    tin: array like
        time as defined.
    Vin: array like
        voltage as defined.
    t : array like
        Time.
    V : array like
        Voltage.
    I : array like
        Current.

    '''
    if action not in ["Forming", 
                      "Sweep", 
                      "Switching with read",
                      "Switching without read"]:
        raise Exception(("Please define a your measurement."))
    
    tin, Vin, t, V, I = read_data(measurement_path, measurement_nr)
    data = bearcats.DataFrame([{ 'device': device_name,
                            #'V': V,
                            #'I': I,
                            #'t': t,
                            'datetime': datetime.now().timestamp(), 
                            "measurement_path": measurement_path,
                            "measurement_nr": measurement_nr,
                            "action": action
                            }])
    
    filename_pkl = f"{action}_{'V_set='}{np.around(max(V[0]),2)}_{'V_V_reset='}{np.around(min(V[0]),2)}_V_{get_formatted_datetime()}.pkl"
    
    data.to_pickle(os.path.join(dir_data_save, filename_pkl))
    return tin, Vin, t, V, I


################################
#  Find read sections
###############################
# Find read sections
def find_read_sections(read_indices, n = 10, min_length = 10):
    '''
    Find sections in which read voltage is applied.

    Parameters
    ----------
    read_indices : array like 
        Boolean vector of where read voltage.
    n : int, optional
        Minimal distance between two reads. The default is 20.
    min_length : int, optional
        Minimal length of read. The default is 50.

    Returns
    -------
    indices_read : list
        List of indices of reads.

    '''
    
    read_indices = np.array(read_indices)
    diff_indices = read_indices[1:]-read_indices[:-1]
    gaps = np.where(diff_indices>n)[0]
    indices_read = np.array_split(read_indices,gaps+1)
    indices_read = [read for read in indices_read if len(read) > min_length]
        
    return indices_read

#####################################################################
## Calculate and save Resistance
#####################################################################    

def calc_R_pulse(I_filt, V_filt, V_read=0.2, tin=[0, 1e-6], Vin=[0.2, 0.2], diff_t=1e-9):
    '''
    Calculate resistance in pulse by absolute amplitude.

    Parameters
    ----------
    I_filt : array like
        (Filtered) current measurement.
    V_filt : array like
        (Filtered) voltage measurement.
    V_read : float, optional
        Read voltage. The default is 0.2.

    Returns
    -------
    R_states : array like
        Resistance of device.

    '''
    R_states, states = [], []
    len_read = (tin[np.where(Vin==0.2)[0][1]]-tin[np.where(Vin==0.2)[0][0]])/(diff_t*5)
    for i, I_f in enumerate(I_filt): 
        segments = find_read_sections(np.where(np.logical_and(V_filt[i]>V_read-0.1,V_filt[i]<V_read+0.1))[0],n=20,min_length=10)
        segments = [seg for seg in segments if len(seg)>len_read and diff_t*2*np.mean(seg)>tin[Vin==0.2][0]] 
        if np.mod(len(segments),2) ==0:  
            state_wf = get_states(Vin, Vread=0.2)
            for j, indices in enumerate(segments):
                indices = indices[int(len(indices)/3):-int(len(indices)/8)]
                
                R_states.append(abs(np.mean(abs(V_filt[i][indices[2:]]/I_filt[i][indices[2:]]))))
                try:
                    states = states + [state_wf[j]]     
                except: 
                    states = states + [state_wf[np.mod(j,2)]]     
    R_states, states = np.array(R_states), np.array(states)
    R_states[R_states>2e5] = 2e5
    return R_states, states

def calc_R_sweep(I_filt, V_filt, V_read=0.2, sr=1e3):
    '''
    Calculate resistance in sweep by slope.

    Parameters
    ----------
    I_filt : array like
        (Filtered) current measurement.
    V_filt : array like
        (Filtered) voltage measurement.
    V_read : float, optional
        Read voltage. The default is 0.2.

    Returns
    -------
    R_states : array like
        Resistance of device.

    '''
    if sr==1e3:
        parasitic_slope=3.8e-6
    elif sr==1e4:
        parasitic_slope=4.5e-6
    else:
        parasitic_slope=7.035e-6

    R_states, states = [], []
    for index_f, V_f in enumerate(V_filt):
        for index_sec, indices in enumerate(find_read_sections(np.where(np.logical_and(abs(V_f)>V_read-0.1,
                                                                abs(V_f)<V_read+0.1))[0], n=20)):
            if (abs(V_f[indices[0]]-V_f[indices[-1]])>0.1) and (V_f[indices[-1]]<-0.05):
                
                # There are different ways of calculating the resistnace
                # Currently we prefer fitting to the Gerade, as we hope  to mitigate outlieres by this in the best way
                state = "HRS" if abs(V_f[indices[0]])-abs(V_f[indices[-1]])>0.1 else "LRS"
                if False:
                    R_states.append(abs(V_f[indices][0]-V_f[indices][-1])/abs(I_filt[index_f][indices[-1]])-I_filt[index_f][indices[0]])
                    
                else:
                    pol_fit = np.polyfit(V_f[indices], I_filt[index_f][indices], 1)
                    
                    states.append(state)
                    R_states.append(abs(1/(abs(pol_fit[0])-abs(parasitic_slope))))
                    
    R_states, states = np.array(R_states), np.array(states)
    R_states[R_states>2e5] = 2e5
    return R_states, states


#####################################################################
## Combined Eval. fct.
#####################################################################    

def main_eval(dir_device: str, 
              measurement_path: str, 
              measurement_nr: int, 
              action: str, 
              device_name: str, 
              bool_sweep=True,
              df_endurance=None):
    '''
    Use all functions for evaluating the data

    Parameters
    ----------
    dir_device : str
        Path for storing the data.
    measurement_path : str
        Path to hdf5 file for rawdata.
    measurement_nr : int
        Number of measurement.
    action : str
        Kind of measurements.
    id_site : str
        Name of device site.
    bool_sweep : bool, optional
        Sweep or not. The default is True.
    df_endureance : DataFrame, optional
        Dataframe, in which the resistance is stored. 
        If Pandas df is received, the new R values are added. Otherwise, 
        a new dataframe is initialized. 

    Returns
    -------
    R_states : list
        Resistance.

    '''
    tin, Vin, t, V, I = load_raw_data_and_store(measurement_path, 
                                                measurement_nr,
                                                action,
                                                device_name,
                                                dir_data_save=dir_device)
    
    I_filt, V_filt, t_filt = filter_data(I, V, t)
        
    if bool_sweep: 
        R_states, states = calc_R_sweep(I_filt, V_filt, V_read=0.2)
    else: 
        R_states, states = calc_R_pulse(I_filt, V_filt, V_read=0.2,tin=tin, Vin=Vin, diff_t=np.diff(t_filt[0][:100])[-1])
    
    if type(df_endurance) == type(None):
        df_endurance = bearcats.DataFrame([{'device': device_name,
                                'R': R_states,
                                "state": states,
                                'datetime': datetime.now().timestamp(), 
                                "measurement_path": measurement_path,
                                "measurement_nr": measurement_nr,
                                "action": action
                                }])
    else: 
        df_endurance = bearcats.concat([df_endurance, 
                                        bearcats.DataFrame([{'device': device_name,
                                       'R': R_states,
                                       "state": states,
                                        'datetime': datetime.now().timestamp(), 
                                        "measurement_path": measurement_path,
                                        "measurement_nr": measurement_nr,
                                        "action": action
                                        }])], ignore_index = True)
    
    # Make figure
    make_figures(dir_device, action, tin, Vin, t, V, I, t_filt, V_filt , I_filt)
    
    return R_states, df_endurance, states