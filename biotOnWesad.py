try:
    from biot.BIOT.run_binary_supervised import Supervised
    from biot.BIOT.run_multiclass_supervised import Supervised as Supervised2
except ModuleNotFoundError:
    from run_binary_supervised import Supervised
    from run_multiclass_supervised import Supervised as Supervised2
# from inputimeout import inputimeout, TimeoutOccurred
import json
import time
import argparse

class Args:
    def __init__(self, test=4):
        self.epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 16
        self.num_workers = 2
        self.dataset = "WESAD"
        self.model = "BIOT"
        self.in_channels = 16
        self.sample_length = 10
        self.n_classes = 2
        self.sampling_rate = 200
        self.token_size = 200
        self.hop_length = 100
        self.test = test
        self.step_size = 240
        self.window_size = 24
        self.sensors = [
            # 'wrist_ACC',
            'wrist_BVP',
            'wrist_EDA',
            'wrist_TEMP',
            # 'chest_ACC',
            # 'chest_ECG',
            # 'chest_EMG',
            # 'chest_EDA',
            # 'chest_Temp',
            # 'chest_Resp',
        ]
        self.imbalance_dels = 0
        self.feat_meth = "resample"
        # self.set_server("kaggle")
    def set_server(self, server):
        if server not in ['pc', 'colab', 'kaggle']:
            raise ValueError("Server not defined")
        self.server = server
        print(f"server set as {server}")
        if self.server == "pc":
            self.pretrain_model_path = r""
        elif self.server == 'kaggle':
            self.pretrain_model_path = r"/kaggle/input/wesad-emotion-dataset/pretrained-models/EEG-PREST-16-channels.ckpt"
        elif self.server == 'colab':
            self.pretrain_model_path = r'pretrained-models/EEG-PREST-16-channels.ckpt'
        
        
        if server == "kaggle":
            self.logpath = "/kaggle/working/log.txt"
        else:
            self.logpath = 'log.txt'
        
        if self.server in ['kaggle', 'colab']:
            self.device = "cuda"
        else:
            self.device = 'cpu'
        
        
    
    def update_by_parser_args(self, args):
        if args.sensors == "all":
            self.sensors = ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC', 'chest_ECG', 'chest_EMG', 'chest_EDA', 'chest_Temp', 'chest_Resp',]
        elif args.sensors == 'all_wrist':
            self.sensors = ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP',]
        elif args.sensors == 'all_chest':
            self.sensors = ['chest_ACC', 'chest_ECG', 'chest_EMG', 'chest_EDA', 'chest_Temp', 'chest_Resp',]
        elif args.sensors == "all_wrist_nacc":
            self.sensors = ['wrist_BVP', 'wrist_EDA', 'wrist_TEMP',]
        elif args.sensors == "all_chest_nacc":
            self.sensors = ['chest_ECG', 'chest_EMG', 'chest_EDA', 'chest_Temp', 'chest_Resp',]
        elif args.sensors == "all_nacc":
            self.sensors = ['wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ECG', 'chest_EMG', 'chest_EDA', 'chest_Temp', 'chest_Resp',]
        else:
            self.sensors = [args.sensors]
        
        if args.stepsize != -1:
            self.step_size = args.stepsize
        if args.windowsize != -1:
            self.window_size = args.windowsize
        self.n_classes = args.nclasses
        self.feat_meth = args.featmeth
        self.set_server(args.server)
        
        

class Main:
    def __init__(self, tests=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17], manual_args=None) -> None:
        self.tests = tests[manual_args.start:]
        self.args = manual_args
        self.logpath = "/kaggle/working/log.txt" if manual_args.server == 'kaggle' else 'log.txt'

    def go(self):
        with open(self.logpath, "w") as file:
            file.write("")    
        results = {}
        for test in self.tests:
            print(f"\033[93mTesting {test}\033[0m")
            start_time = time.time()
            args = Args(test=test)

            if self.args:
                args.update_by_parser_args(self.args)
            
            model = Supervised2(args=args)
            results[test] = model.supervised_go()
            del model

            total_time = time.time() - start_time
            results[test]['time'] = total_time
            print(f"\033[92mTotal time: {total_time} secs\033[0m")
            # cal total
            mean_results = {
                metric: sum([results[subj][metric] for subj in results.keys()]) / (len(results.keys()) if 'total' not in results else len(results.keys()) - 1)
                for metric in results[list(results.keys())[0]]
            }
            results["total"] = mean_results
            
            with open("/".join(self.logpath.split("/")[:-1]) + f"/biot_result_{time.strftime('%Y_%b_%d_%H')}.json", "w") as file:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", type=str, default="all", required=False)
    parser.add_argument("--stepsize", type=int, default=-1, required=False)
    parser.add_argument("--windowsize", type=int, default=-1, required=False)
    parser.add_argument("--nclasses", type=int, default=2, required=False)
    parser.add_argument("--featmeth", type=str, default="resample", required=False)
    parser.add_argument("--server", type=str, default="kaggle", required=False)
    parser.add_argument("--start", type=int, default=0, required=False)
    manual_args = parser.parse_args()
    print(manual_args._get_kwargs())

    runner = Main(manual_args=manual_args)
    runner.go()