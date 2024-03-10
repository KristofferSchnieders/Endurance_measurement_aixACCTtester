# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:51:42 2024

@author: schnieders
"""
import numpy as np

#####################################################################
## Wafermaps
#####################################################################

def wafermap_Neurotech1_1R(): 
    '''
    Currently dummy

    Returns
    -------
    position : array like
        Positions of devices.
    name : array like
        Name of devices.
    geometry : array like
        Geometry. (For 1R probably not necessary)

    '''
    
    '''
    Wafermap for the 1R devices on the X-Fab chip. We devide the devices in three
    different categories. 
        1. The devices in the vias left and right to the array blocks. 
        2. The devices in the array blocks with normal spacing.
        3. The devices in the array blocks with different spacing. 
           (irregular y-distance).
    '''
    
    position_devices  = [np.array([0,0])]
    name_device= [["str", "dev"]]
    geometries = [["100", "100"]]
    # Spacing between devices
    dist_1R_array =  np.array([130, -290])
    dist_1R_array_special =  np.array([130, -350])
    dist_1R_track =  np.array([0, -290])


    # Number devices in structure
    nr_device_in_track = [[22], [21], [21], [22]]
    nr_devices_in_line_array = [[16, 16, 16, 16, 16, 16], 
                                [16, 17, 17, 17, 17, 17], 
                                [13, 13, 13, 13, 13,  0]]
    nr_devices_in_line_array_special = [[14], [11], [5]]

    # Distances between reference device and first device in structure. 
    vector_reference_device = -np.array([[0, 0], 
                                        [-660, 500], 
                                        [-530, 790], 
                                        [-1070, 3870], 
                                        [-550, 4160],
                                        [-4040, 125],
                                        [-4170, 125],
                                        [-4750, 760],
                                        [-4750, 1110],
                                        [-7300, 0]])

    names_structure = ["BT3",
                       "Array_1",
                       "Array_1",
                       "Array_2", 
                       "Array_2",
                       "D7",
                       "D8",
                       "Array_3",
                       "Array_3",
                       "AR1"]
    
    geom_str = [['100', "100"],
                    ['100', "100"],
                    ['100', "100"],
                    ['100', "100"],
                    ['100', "100"],
                    ['100', "100"],
                    ['100', "100"],
                    ['100', "100"],
                    ['100', "100"],
                    ['100', "100"]]

    name_row, name_column = list(range(1,31)), ["a","b","c","d","e","f","g","h","i","k",
                                                "l","m","n","o","p","q","r","s","t","u",
                                                "v","w","x","y","z"]
    
    for nr_struct, dist_struct in enumerate(vector_reference_device):
        if nr_struct in [0, 5, 6, 9]:

            dist_next_device = dist_1R_track * np.array([0, 1])
            dist_next_line   = dist_1R_track * np.array([1,0])
            nr_in_lines = nr_device_in_track.pop(0)
    
        elif nr_struct in [1, 3, 7]: 

            dist_next_device = dist_1R_array_special * np.array([1, 0])
            dist_next_line   = dist_1R_array_special * np.array([0, 1])
            nr_in_lines = nr_devices_in_line_array_special.pop(0)
    
        elif nr_struct in [2, 4, 8]:

            dist_next_device = dist_1R_array * np.array([1, 0])
            dist_next_line   = dist_1R_array * np.array([0, 1])
            nr_in_lines = nr_devices_in_line_array.pop(0)
        
        for index_y, nr_devices_in_line in enumerate(nr_in_lines):
            for index_x in range(nr_devices_in_line):
                position_devices.append(dist_struct + 
                        index_x * dist_next_device +
                        index_y * dist_next_line)
                geometries.append(geom_str[nr_struct])
        
        name_str = names_structure[nr_struct]

        if nr_struct in [0, 5, 6, 9]:
            for index_y, nr_devices_in_line in enumerate(nr_in_lines):
                for index_x in range(nr_devices_in_line):
                    name_instr = name_column[index_y] + str(name_row[index_x])
                    name_device.append([names_structure[nr_struct], name_instr])
    
        elif nr_struct in [1, 3, 7]: 

            for index_y, nr_devices_in_line in enumerate(nr_in_lines):
                for index_x in range(nr_devices_in_line):
                    if nr_struct==1:
                        offset_x = 1
                    elif nr_struct==3:
                        offset_x = 4
                    else:
                        offset_x = 0
                        
                    name_instr = name_column[index_y] + str(name_row[index_x+offset_x])
                    name_device.append([names_structure[nr_struct], name_instr])
        elif nr_struct in [2, 4, 8]:

            for index_y, nr_devices_in_line in enumerate(nr_in_lines):
                for index_x in range(nr_devices_in_line):
                    name_instr = name_column[index_y+1] + str(name_row[index_x])
                    name_device.append([names_structure[nr_struct], name_instr])


    ##############################################################################
    ## Bring data to readable form.
    ##############################################################################
    position_devices.pop(0), name_device.pop(0), geometries.pop(0)
    positions, name, geometry =-np.array(position_devices), np.array(name_device), np.array(geometries)

    return positions, name, geometry

def wafermap_Neurotech1_1T1R():
    '''
    Currently dummy

    Returns
    -------
    position : array like
        Positions of devices.
    name : array like
        Name of devices.
    geometry : array like
        Dimensions of transistors.

    '''
    '''
    Wafermaps for the 1T1R devices on the X-Fab chip. We transistors in horizontal 
    and vertical direction. All 1T1R devices are in vias of different length. 
    In vertical direction: 8 long vias 
    In horizontal direction: 14 vias of which 12 are short because they are 
                             interrupted by the vertical structurs.
    '''
    position_devices  = [np.array([0,0])]
    # Spacing between devices
    dist_1T1R_vias =  np.array([0, 610])


    #############################################################################
    ## vertical 1T1R structures
    #############################################################################

    position_devices_vertical  = [np.array([0,0])]
    transistor_type_vertical = ['']
    tracks_vertical = ['']

    names_traces_vertiacal = ["A_LLL", "D_LLLL", 
                              "D_LLL", "D_LL", 
                              "D_L", "D_R",
                              "D_RR", "A_RRR"]
    nr_device_in_vias_vertical = [[11], [10], 
                                   [10], [10], [10], [10], [10], [11]]
    name_transistors_vertical = [['1T1R_ne5_u220_u500'], ['1T1R_ne5_u220_u500'], 
                                 ['1T1R_ne5_10u_u500'], ['1T1R_ne5_u500_10u'],  
                                 ['1T1R_ne5_10u_10u'],['1T1R_ne5_u220_u500'],
                                ['1T1R_ne5_u500_10u'], ['1T1R_ne5_u500_10u']]

    vector_reference_device_vertical = np.array([[0, 0], 
                                        [-3390, 290], 
                                        [-3520, 290], 
                                        [-3650, 290],
                                        [-3780, 290],
                                        [-4040, 290],
                                        [-4170, 290],
                                        [-7820, 0]])


    for nr_struct, dist_struct in enumerate(vector_reference_device_vertical):
        dist_next_device = dist_1T1R_vias * np.array([0, 1])
        nr_in_via_vertical = nr_device_in_vias_vertical[nr_struct][0]
        
        for index_x, nr_devices_in_line in enumerate(range(nr_in_via_vertical)):
            position_devices_vertical.append(dist_struct + 
                    index_x * dist_next_device)
            transistor_type_vertical.append(
                name_transistors_vertical[nr_struct][0])
            tracks_vertical.append(names_traces_vertiacal[nr_struct]+ "-" + \
                                     str(nr_devices_in_line))
    ##############################################################################
    ## Bring data to readable form.
    ##############################################################################
    position_devices_vertical.pop(0)
    transistor_type_vertical.pop(0)
    tracks_vertical.pop(0)
    position, name, geometry = np.array(position_devices_vertical), np.array(tracks_vertical), np.array(transistor_type_vertical)
    return position, name, geometry

def wafermap_Neurotech1_1T1R_horizontal():
    '''
    Currently dummy

    Returns
    -------
    position : array like
        Positions of devices.
    name : array like
        Name of devices.
    geometry : array like
        Dimensions of transistors.

    '''
    '''
    Wafermaps for the 1T1R devices on the X-Fab chip. We transistors in horizontal 
    and vertical direction. All 1T1R devices are in vias of different length. 
    In vertical direction: 8 long vias 
    In horizontal direction: 14 vias of which 12 are short because they are 
                             interrupted by the vertical structurs.
    '''

    position_devices_horizontal  = [np.array([0,0])]
    transistor_type_horizontal = ['']
    tracks_horizontal = ['']
    
    names_traces_horizontal = ["B_BBB", "B_TTT", 
                              "C_LT", "C_L", "C_LB", "C_RT", "C_R", "C_RB",
                              "C_TTTX", "C_TTX", "C_TX", 
                              "C_BBBX", "C_BBX", "C_BX"]
    nr_device_in_vias_horizontal = [[11], [11], 
                                   [4], [4], [4], [4], [4], [4],
                                   [4], [4], [4], 
                                   [4], [4], [4]]
    name_transistors_horizontal = [['1T1R_ne5_u500_10u'], ['1T1R_ne5_10u_10u'],
                     ['11T1R_ne5_10u_u500'], ['1T1R_ne5_10u_u500'],
                     ['11T1R_ne5_10u_u500'],  ['1T1R_ne5_10u_10u'],
                     ['1T1R_ne5_10u_10u'], ['1T1R_ne5_10u_10u'],
                     ['1T1R_ne5_10u_u500'], ['1T1R_ne5_10u_u500'],
                     ['1T1R_ne5_10u_u500'],  ['1T1R_ne5_10u_10u'],
                     ['1T1R_ne5_10u_10u'], ['1T1R_ne5_10u_10u']]
    
    vector_reference_device_horizontal = np.array([[0, 0], 
                                        [-7040, 25], 
                                        [-3650, 115], 
                                        [-3520, 115], 
                                        [-3390, 115],
                                        [-3650, 4355],
                                        [-3520, 4355],
                                        [-3390, 4355], 
                                        [-7470, 2215], 
                                        [-7340, 2215], 
                                        [-7210, 2215],
                                        [400, 2215],
                                        [270, 2215],
                                        [140, 2215]])
    
    for nr_struct, dist_struct in enumerate(vector_reference_device_horizontal):
        dist_next_device = dist_1T1R_vias * np.array([0, 1])
        nr_in_via_horizontal = nr_device_in_vias_horizontal[nr_struct][0]
        
        for index_x, nr_devices_in_line in enumerate(range(nr_in_via_horizontal)):
            position_devices_horizontal.append(dist_struct + 
                    index_x * dist_next_device)
            transistor_type_horizontal.append(
                name_transistors_horizontal[nr_struct][0])
            tracks_horizontal.append(names_traces_horizontal[nr_struct] + "-" + \
                                     str(nr_devices_in_line))
    ##############################################################################
    ## Bring data to readable form.
    ##############################################################################
    position_devices_horizontal.pop(0)
    transistor_type_horizontal.pop(0)
    tracks_horizontal.pop(0)
    position, name, geometry = np.array(position_devices_horizontal), np.array(tracks_horizontal), np.array(transistor_type_horizontal)
    return position, name, geometry
