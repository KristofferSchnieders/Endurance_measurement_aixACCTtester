# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:07:56 2024

@author: schnieders
"""

import numpy as np 
import h5py
import datetime
import pandas as bearcats
import os


# Read the measured data
def read_data(measurement_path: str, measurement_nr: int):
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
    with h5py.File(measurement_path ,"r") as f:
        tin =np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_DA_wedge02"][0])
        Vin =np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_DA_wedge02"][1])
        tread = np.array([t[0] for t in np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_AD_wedge02"])])
        Vread = np.array([I[1] for I in np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_AD_wedge02"])])
        Iread = np.array([V[1] for V in np.array(f[f"Container_0/DataSet_{measurement_nr}/MatrixDouble_AD_wedge03"])])
 
    return tin, Vin, tread, Vread, Iread

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
    I_filt, V_filt, t_filt = list(), list(), list()
    for i, dummmy_I in enumerate(I):
        I_offset = np.mean(I[i][-100:])
        I_filt_dummy=running_median(I[i]-I_offset)
        I_filt=I_filt.append((np.cumsum(I_filt_dummy[Nmean:])-np.cumsum(I_filt_dummy[:-Nmean]))/Nmean)
        
        V_offset = np.mean(V[i][-100:])
        V_filt_dummy=running_median(V[i]-V_offset)
        V_filt=V_filt.append((np.cumsum(V_filt_dummy[Nmean:])-np.cumsum(V_filt_dummy[:-Nmean]))/Nmean)
        
        t_filt=t_filt.append(t[i][3:len(I_filt_dummy)+3])

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
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M')
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
    id_site : int
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
                            'V': V,
                            'I': I,
                            't': t,
                            'datetime': datetime.now().timestamp(), 
                            "measurement_path": measurement_path,
                            "measurement_nr": measurement_nr,
                            "action": action
                                }])

    filename_pkl = f"{action}_{'V_set='}{np.around(max(V),2)}_{'V_reset='}{np.around(min(V),2)}_{get_formatted_datetime()}.pkl"
    
    data.to_pickle(os.path.join(dir_data_save, filename_pkl))
    return tin, Vin, t, V, I


################################
#  Find read sections
###############################
# Find read sections
def find_read_sections(read_indices, n = 20, min_length = 50):
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

def calc_R_pulse(I_filt, V_filt, V_read=0.2):
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
    R_states = []
    for i in I_filt: 
        for indices in find_read_sections(np.where(np.logical_and(abs(V_filt[i])>V_read-0.05,
                                                                  abs(V_filt[i])<V_read+0.05))[0]):
            R_states = R_states.append(np.mean(V_filt[i][indices]/I_filt[i][indices]))
    return R_states

def calc_R_sweep(I_filt, V_filt, V_read=0.2):
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

    R_states = []
    for indices in find_read_sections(np.where(np.logical_and(abs(V_filt)>V_read-0.1,
                                                              abs(V_filt)<V_read+0.1))[0]):
        R_states = R_states.append(abs(max(V_filt[indices])-min(V_filt[indices]))/abs(max(I_filt[indices])-min(I_filt[indices])))
    return R_states

#####################################################################
## Combined Eval. fct.
#####################################################################    

def main_eval(dir_device: str, 
              measurement_path: str, 
              measurement_nr: int, 
              action: str, 
              device_name: str, 
              bool_sweep=True):
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
        R_states = calc_R_sweep(I_filt, V_filt, V_read=0.2)
    else: 
        R_states = calc_R_pulse(I_filt, V_filt, V_read=0.2)
    
    return R_states