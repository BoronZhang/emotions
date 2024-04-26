import numpy as np
import pandas as pd

# Clean Sx_E4_Data
for file in ['BVP.csv', 'EDA.csv', 'HR.csv', 'TEMP.csv']:
    df = pd.read_csv('WESAD/S2/S2_E4_Data/' + file)
    freq = df.loc[0].item()
    start = 640 if file == 'HR.csv' else 0
    times = [start + i * (64 // freq) for i in range(len(df) - 1)]
    df_new = pd.DataFrame({'time': times, 'data': df.loc[1:, df.columns[0]]})
    df_new['time'] = df_new['time'].astype(np.int32)
    df_new.set_index('time', inplace=True)
    df_new.to_csv('WESAD/S2/S2_E4_Data_new/' + file)

for file in ['ACC.csv']:
    df = pd.read_csv('WESAD/S2/S2_E4_Data/' + file)
    freq = df.loc[0, df.columns[0]].item()
    times = [start + i * (64 // freq) for i in range(len(df) - 1)]
    df_new = pd.DataFrame({'time': times, 
                           'X': df.loc[1:, df.columns[0]], 
                           'Y': df.loc[1:, df.columns[1]], 
                           'Z': df.loc[1:, df.columns[2]], 
                           })
    df_new['time'] = df_new['time'].astype(np.int32)
    df_new.set_index('time', inplace=True)
    df_new.to_csv('WESAD/S2/S2_E4_Data_new/' + file)




    
