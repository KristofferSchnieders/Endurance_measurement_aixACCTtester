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

def bool_switched(R_LRS, R_HRS, bool_LRS, bool_HRS): 
    return np.logical_and(bool_LRS(R_LRS), bool_HRS(R_HRS))

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