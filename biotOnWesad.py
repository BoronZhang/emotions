from biot.BIOT.run_binary_supervised import Supervised
from biot.BIOT.run_multiclass_supervised import Supervised as Supervised2
# from inputimeout import inputimeout, TimeoutOccurred
import json
import time

class Args:
    def __init__(self, test=4):
        self.epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.num_workers = 4
        self.dataset = "WESAD"
        self.model = "BIOT"
        self.in_channels = 16
        self.sample_length = 10
        self.n_classes = 3
        self.sampling_rate = 200
        self.token_size = 200
        self.hop_length = 100
        self.server = "kaggle" # or colab
        if self.server == "pc":
            self.pretrain_model_path = r"C:/Users/Elham moin/Desktop/uniVer/bachProj/biot/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
        elif self.server == 'kaggle':
            self.pretrain_model_path = r"/kaggle/input/wesad-emotion-dataset/biot/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
        elif self.server == 'colab':
            self.pretrain_model_path = r'/biot/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt'
        self.test = test
        if self.server in ['kaggle', 'colab']:
            self.device = "gpu"
        else:
            self.device = 'cpu'
        self.step_size = 240
        self.window_size = 240
        self.sensors = [
            'wrist_ACC',
            'wrist_BVP',
            'wrist_EDA',
            'wrist_TEMP',
            'chest_ACC',
            'chest_ECG',
            'chest_EMG',
            'chest_EDA',
            'chest_Temp',
            'chest_Resp',
        ]
        self.imbalance_dels = 50000
        self.logpath = "/kaggle/working/log.txt"
        
class Main:
    def __init__(self, tests=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17], **kwargs) -> None:
        self.tests = tests
        self.kwargs = kwargs
        self.logpath = "/kaggle/working/log.txt"

    def go(self):
        with open(self.logpath, "w") as file:
            file.write("")    
        results = {}
        for test in self.tests:
            print(f"\033[93mTesting {test}\033[0m")
            start_time = time.time()
            args = Args(test=test)

            for key in self.kwargs:
                if hasattr(args, key):
                    setattr(args, key, self.kwargs[key])
            
            self.model = Supervised2(args=args)
            results[test] = self.model.supervised_go()
            total_time = time.time() - start_time
            results[test]['time'] = total_time
            print(f"\033[92mTotal time: {total_time} secs\033[0m")
            # cal total
            mean_results = {
                metric: sum([results[subj][metric] for subj in results.keys()]) / (len(results.keys()) if 'total' not in results else len(results.keys()) - 1)
                for metric in results[list(results.keys())[0]]
            }
            results["total"] = mean_results
            
            with open(f"biot_result_{time.strftime('%Y_%b_%d_%H')}.json", "w") as file:
                json.dump(results, file, indent=2)
            # try:
            #     go = inputimeout("\033[91mWould you like to continue: \033[0m", 30)
            #     print(f"test = {test}")
            #     if go in ["cancel", "0", "done", "no", "n"]:
            #         break
            # except TimeoutOccurred:
            #     pass
            # except KeyboardInterrupt:
            #     break
    
    
if __name__ == "__main__":
    runner = Main()
    runner.go()