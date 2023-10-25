import numpy as np
import pandas as pd
import os 
import shutil 


def main():
    EU_label = pd.read_csv("EU_label.csv")
    flowid_lst = EU_label["file_flow_id"].to_list()
    healthy_flowid_lst = EU_label.loc[EU_label['label'] == 'Healthy']["file_flow_id"].to_list() 
    sick_flowid_lst = EU_label.loc[EU_label['label'] == 'Sick']["file_flow_id"].to_list()
    
     
    for src in flowid_lst:
        dest = "raw_fcs/" + src
        src_files = os.listdir(dest) 
    
        if src in healthy_flowid_lst:
    
            for file_name in src_files: 
                full_file_name = os.path.join(dest, file_name) 
                # copy fcs files to raw_fcs/ 
                if os.path.isfile(full_file_name): 
                    shutil.copy(full_file_name, "raw_fcs/healthy" )
        else:
            for file_name in src_files: 
                full_file_name = os.path.join(dest, file_name) 
                # copy fcs files to raw_fcs/ 
                if os.path.isfile(full_file_name): 
                    shutil.copy(full_file_name, "raw_fcs/sick" )



if __name__ == "__main__":
    main()