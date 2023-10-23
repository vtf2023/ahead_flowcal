import numpy as np
import pandas as pd
import FlowCal
import matplotlib.pyplot as plt
import os 


marker_channel_map = pd.read_csv("EU_marker_channel_mapping.csv")
channels_use = marker_channel_map.index[marker_channel_map['use']==1].tolist()
channels_names = marker_channel_map.loc[marker_channel_map['use']==1]["PxN(channel)"].tolist()

heal_files = os.listdir("raw_fcs/healthy") 
sick_files = os.listdir("raw_fcs/sick") 

heal =[]
sick =[]

for file_name in heal_files: 
    full_file_name = os.path.join("raw_fcs/healthy", file_name) 
    
    s1 = FlowCal.io.FCSData(full_file_name)
    s2 = s1[:, channels_names]
    s = FlowCal.transform.to_rfi(s2)  # maybe don't need to do it
    s_g1 = FlowCal.gate.high_low(s, channels=channels_names)  # best gating from domain knowledge
    heal.append(np.array(s_g1))

np.save("healthy_g1.npy", heal, allow_pickle=True)

for file_name in sick_files: 
    full_file_name = os.path.join("raw_fcs/sick", file_name) 
    
    s1 = FlowCal.io.FCSData(full_file_name)
    s2 = s1[:, channels_names]
    s = FlowCal.transform.to_rfi(s2)
    s_g1 = FlowCal.gate.high_low(s, channels=channels_names)
    sick.append(np.array(s_g1))

np.save("sick_g1.npy", sick, allow_pickle=True)

