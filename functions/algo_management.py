# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:06:24 2024

@author: schnieders
"""
import numpy as np
import requests

def bool_states(interval_LRS, interval_HRS):
    '''
    Define functions to check, if resistance in HRS or LRS

    Parameters
    ----------
    interval_LRS : array like
        [Lower_limit_LRS, Upper_limit_LRS].
    interval_HRS : array like
        [Lower_limit_HRS, Upper_limit_HRS].

    Returns
    -------
    [0]: Function returning if device in LRS
    [1]: Function returning if device in HRS
    '''
    
    def bool_HRS(R: float):
        '''
        Check HRS

        Parameters
        ----------
        R : Float
            Resistance.

        Returns
        -------
        int
            0: Between states,
            1: In interval,
            2: Undefined states.

        '''
        if R<interval_HRS[0]:
            return 0
        elif R>interval_HRS[0] and R<interval_HRS[1]:
            return 1
        elif R>interval_HRS[1]:
            return 2
        
    def bool_LRS(R: float):
        '''
       Check LRS     

        Parameters
        ----------
        R : Float
            Resistance.

        Returns
        -------
        int
            0: Between states,
            1: In interval,
            2: Undefined states.

        '''
        if R<interval_LRS[0]:
            return 2
        elif R>interval_LRS[0] and R<interval_LRS[1]:
            return 1
        elif R>interval_LRS[1]:
            return 0
    return bool_LRS, bool_HRS

def bool_switched(R, states, bool_LRS, bool_HRS):
    R_HRS, R_LRS = R[states=='HRS'], R[states=='LRS']
    HRS_switched, LRS_switched = np.array([bool_HRS(R) for R in R_HRS]), np.array([bool_LRS(R) for R in R_LRS])
    min_len_switched = np.min([len(HRS_switched), len(LRS_switched)])
    if len(HRS_switched)!=len(LRS_switched):
        print("Have a look at data. The evaluation seems unstable.")
    
    bool_switched = np.logical_and(HRS_switched[:min_len_switched]==1,
                                   LRS_switched[:min_len_switched]==1)
    return bool_switched, [HRS_switched, LRS_switched]

def send_msg(text):
    '''
    Send message over Telegram

    Parameters
    ----------
    text : str
        Text you want to send to yourself.

    Returns
    -------
    None.

    '''
    token = "5697756678:AAEjgVl7IDwsNAUBnBdyn_S1KsqQXWr7ytA"
    chat_id = "848875578"
    url_req = r'https://api.telegram.org/bot' + token + "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text
    results = requests.get(url_req)
    print(results.json())

def choose_gate_V(V_gate, geomet):
    if geomet in "1T1R_ne5_u500_10u":
        return V_gate["u500_10u"], False
    elif geomet in "1T1R_ne5_u220_u500":
        return V_gate["u500_10u"], True
    elif geomet in "1T1R_ne5_10u_10u":
        return V_gate["10u_10u"], True
    elif geomet in "1T1R_ne5_10u_u500":
        return V_gate["10u_u500"],True