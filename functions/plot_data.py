# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:50:14 2024

@author: schnieders
"""

# Mathematical functions
import numpy as np

# Plotting 
import matplotlib.pyplot as plt

# Handling of foldersystem
import os
import sys 

sys.path.append(r"D:\Scripts\Schnieders\Endurance_measurement_aixACCTtester\functions")

from data_management import *


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

def make_figures(dir_device, action, tin, Vin, t, V, I, t_filt, V_filt , I_filt):
    I_offset=np.mean(I[-1])
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(tin, Vin)
    ax[0,0].set_title(' Voltage in')
    ax[0,1].plot(t[-1], V[-1])
    ax[0,1].plot(t[-1],(I[-1]-I_offset)/max((I[-1]-I_offset))*max(V[-1])*0.6)
    ax[0,1].set_title('Voltage monitored, Filt. Current')
    ax[1,0].plot(t[-1], I[-1])
    ax[1,0].set_title('Current')
    ax[1,1].plot(t_filt[-1],I_filt[-1])
    ax[1,1].set_title('Filtered Current')
    fig.set_figheight(6)
    fig.set_figwidth(9)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_device,  f"{action}_{'V_set='}{np.around(max(Vin),2)}_{'V_reset='}{np.around(min(Vin),2)}_{get_formatted_datetime()}.png"))
    plt.close(fig)
    
def figure_endurance(df_endurance, states, device, dir_device):
    R, states = np.concatenate(df_endurance.R),  np.concatenate(df_endurance.state)
    R_HRS, R_LRS = R[states=='HRS'], R[states=='LRS']
    
    
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0,len(R_LRS))+1, R_LRS, color="blue", label=r"R$_{LRS}$")
    ax.scatter(np.arange(0,len(R_HRS))+1, R_HRS, color="red", label=r"R$_{HRS}$")
    ax.set_xlabel("Nr. switched")
    ax.set_ylabel(r"Resistance / $\Omega$")
    ax.set_title(r"Results endurance "+ device)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(dir_device,  f"Endurance_device_{device}_{get_formatted_datetime()}.png"))
    plt.close(fig)