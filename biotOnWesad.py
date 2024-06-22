from biot.BIOT.run_binary_supervised import Supervised
from inputimeout import inputimeout, TimeoutOccurred
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
        self.n_classes = 1
        self.sampling_rate = 200
        self.token_size = 200
        self.hop_length = 100
        self.pretrain_model_path = r"C:\Users\Elham moin\Desktop\uniVer\bachProj\biot\BIOT\pretrained-models\EEG-PREST-16-channels.ckpt"
        self.server = "pc" # or colab
        self.test = test
        self.device = "cpu"
        self.step_size = 240
        self.window_size = 240
        self.common_shape = 12000

class Main:
    def __init__(self, tests=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17], **kwargs) -> None:
        self.tests = tests
        self.kwargs = kwargs

    def go(self):
        with open("log.txt", "w") as file:
            file.write("")    
        results = {}
        for test in self.tests:
            print(f"\033[93mTesting {test}\033[0m")
            start_time = time.time()
            args = Args(test=test)

            for key in self.kwargs:
                if hasattr(args, key):
                    setattr(args, key, self.kwargs[key])
            self.model = Supervised(args=args)
            results[test] = self.model.supervised_go()
            total_time = time.time() - start_time
            results[test]['time'] = total_time
            print(f"\033[92mTotal time: {total_time} secs\033[0m")
            with open(f"biot_result_{time.strftime('%Y_%b_%d_%H')}.json", "w") as file:
                json.dump(results, file, indent=2)
            try:
                go = inputimeout("\033[91mWould you like to continue: \033[0m", 30)
                print(f"test = {test}")
                if go in ["cancel", "0", "done", "no", "n"]:
                    break
            except TimeoutOccurred:
                pass
            except KeyboardInterrupt:
                break
    
    
if __name__ == "__main__":
    runner = Main()
    runner.go()