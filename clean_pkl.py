import pickle
import numpy as np

def clean_pkl():
    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    freqs = {'chest': {sensor: 700 for sensor in {'ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp'}},
            'wrist': {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}}
    total = {}
    for id in subject_ids:
        print(end=f'Doing S{id}')
        with open(f"WESAD/S{id}/S{id}.pkl", 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            labels:np.ndarray = data[b'label']
            signals:np.ndarray = data[b'signal']
            labels_sec = labels.reshape((-1, 700))
            tmp_data = {device: 
                        {sensor.decode(): 
                         signals[device.encode()][sensor].reshape((-1, 3 if sensor == b'ACC' else 1, freqs[device][sensor.decode()]))
                           for sensor in signals[device.encode()]} 
                        for device in ['wrist', 'chest']}
            goods = (labels_sec.min(axis=1) > 0) & (labels_sec.max(axis=1) < 5)
            labels_sec_good = labels_sec[goods]
            tmp_data = {device: {
                sensor: tmp_data[device][sensor][goods] for sensor in tmp_data[device]
            } for device in tmp_data}
            
        tmp_data['label'] = labels_sec_good
        print(end=f'\rWriting S{id}')
        with open(f"WESAD/S{id}/S{id}_n0.pkl", 'wb') as file:
            pickle.dump(tmp_data, file)
        total[f'S{id}'] = tmp_data
        print(f'\rS{id} Finished ({labels.shape[0] / 700} -> {len(labels_sec_good)} seconds)')
    with open(f'WESAD/total.pkl', 'wb') as file:
        pickle.dump(total, file)

clean_pkl()
