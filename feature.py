from collections import defaultdict
import pickle
import os
import numpy as np
import pandas as pd

def extract_features(verbose:int=0, log_file=True, window_size=1):
    # Creating dataframe of all of them
    # `window` | `mean` | `std` | `max` | `min` | `label`
    LABELS = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}
    freqs = {'chest': {sensor: 700 for sensor in {'ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp'}},
            'wrist': {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}}

    window_size = window_size # in second
    def infer_label(labels:np.ndarray):
        counts = defaultdict(lambda: 0)
        for item in labels:
            counts[item] += 1
        max_key, max_value = 0, 0
        for item in counts:
            if counts[item] > max_value:
                max_key, max_value = item, counts[item]
        return max_key

    if log_file:log:str = ""
    for subject in range(2, 18):
        if subject == 12:
            continue
        with open(f'WESAD/S{subject}/S{subject}.pkl', 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='bytes')
        if verbose:print(f"Data of S{subject} loaded!")
        if log_file:log += f"Data of S{subject} loaded!\n"
        os.makedirs(f"WESAD/S{subject}/data", exist_ok=True)

        for bdevice in data[b'signal'].keys():
            for bsensor in data[b'signal'][bdevice]:
                device = bdevice.decode()
                sensor = bsensor.decode()
                df_dict = {'window': [], 'mean': [], 'std': [], 'min': [], 'max': [], 'label': []}
                freq = freqs[device][sensor] * window_size
                raw_data:np.ndarray = data[b'signal'][bdevice][bsensor]
                raw_data = raw_data.reshape((len(raw_data)//freq, freq, -1))
                i = 0
                for window in raw_data:
                    # window is a np.ndarray of the desired size
                    df_dict['window'] = i
                    seconds_elapsed = i * window_size * 700
                    i += 1
                    labels = data[b'label'][seconds_elapsed:seconds_elapsed + window_size * 700]
                    label = infer_label(labels)
                    if label not in LABELS.keys():
                        continue # it would be same for all sets
                    df_dict['label'].append(label)
                    for feature in ['mean', 'std', 'max', 'min']:
                        df_dict[feature].append(getattr(window, feature)(axis=0).sum())
                    if verbose > 2:
                        print(end=f'\r{device}_{sensor}-> {i}/{len(raw_data)}')
                        if log_file:
                            log += f'\r{device}_{sensor}-> {i}/{len(raw_data)}\n'

                df = pd.DataFrame(df_dict)
                df.set_index('window')
                df.to_csv(f'WESAD/S{subject}/data/{device}_{sensor}.csv', index=False)
                if verbose > 1:print(f'{device}_{sensor}-> {df.shape}')
                if log_file:log += f'{device}_{sensor}-> {df.shape}\n'
                

    if log_file:
        with open ('log.txt', 'w') as logFile:
            logFile.write(log)    
            

