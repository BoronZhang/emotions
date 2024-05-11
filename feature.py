from collections import defaultdict
import pickle
import os
import numpy as np
import pandas as pd
from typing import Literal

def extract_features(verbose:int=0, log_file=True, window_size=1, mode:Literal["raw", "manual", "svd"]="raw"):
    # Creating dataframe of all of them
    # `window` | `mean` | `std` | `max` | `min` | `label`
    LABELS = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}
    freqs = {'chest': {sensor: 700 for sensor in {'ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp'}},
            'wrist': {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}}
    MIN_FREQ = 4
    

    def infer_label(labels:np.ndarray) -> int:
        counts = defaultdict(lambda: 0)
        for item in labels:
            counts[item] += 1
        max_key = max(counts, key=lambda x: counts[x])
        return max_key

    log:str = ""
    for subject in range(2, 18):
        if subject == 12:
            continue
        with open(f'WESAD/S{subject}/S{subject}.pkl', 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='bytes')
        if verbose:print(f"Data of S{subject} loaded!")
        if log_file:log += f"Data of S{subject} loaded!\n"
        os.makedirs(f"WESAD/S{subject}/{mode}_data", exist_ok=True)
        

        for bdevice in data[b'signal'].keys():
            for bsensor in data[b'signal'][bdevice]:
                device = bdevice.decode()
                sensor = bsensor.decode()
                if mode in ["raw", "svd"]:
                    df_dict:dict[str, list[float]] = {f"{i}": [] for i in range(MIN_FREQ)}
                    df_dict['window'] = []
                    df_dict['label'] = []
                elif mode == "manual":
                    df_dict:dict[str, list[float]] = {'window': [], 'mean': [], 'std': [], 'min': [], 'max': [], 'label': []}
                freq = freqs[device][sensor] * window_size
                raw_data:np.ndarray = data[b'signal'][bdevice][bsensor]
                raw_data = raw_data.reshape((len(raw_data)//freq, freq, -1))
                i = 0
                for window in raw_data:
                    # window is a np.ndarray of (window_size * freq, features), features is 3 for ACC and 1 for others
                    seconds_elapsed = i * window_size * 700 # for labels
                    i += 1
                    labels = data[b'label'][seconds_elapsed:seconds_elapsed + window_size * 700]
                    label = infer_label(labels)
                    if label not in LABELS.keys():
                        continue # it would be same for all sets
                    df_dict['window'].append(i)
                    df_dict['label'].append(label)

                    if mode == "raw":
                        downsampled = window[::len(window) // MIN_FREQ]
                        for second in range(MIN_FREQ):
                            df_dict[f"{second}"].append(downsampled[second].mean())
                    elif mode == "manual":
                        for feature in ['mean', 'std', 'max', 'min']:
                            df_dict[feature].append(getattr(window, feature)(axis=0).sum())
                    elif mode == "svd":
                        U, s, Vt = np.linalg.svd(window.reshape(4, -1), full_matrices=False)
                        pca = (U[:, :1] * s[:1]).squeeze()
                        for second in range(MIN_FREQ):
                            df_dict[f"{second}"].append(pca[second].mean())
                    if verbose > 2:
                        print(end=f'\r{device}_{sensor}-> {i}/{len(raw_data)}')
                        if log_file:
                            log += f'\r{device}_{sensor}-> {i}/{len(raw_data)}\n'
                
                df = pd.DataFrame(df_dict)
                df.set_index('window')
                df.to_csv(f'WESAD/S{subject}/{mode}_data/{device}_{sensor}.csv', index=False)
                if verbose > 1:print(f'{device}_{sensor}-> {df.shape}')
                if log_file:log += f'{device}_{sensor}-> {df.shape}\n'
                

    if log_file:
        with open ('log.txt', 'w') as logFile:
            logFile.write(log)    
            

