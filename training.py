from sklearn.base import ClassifierMixin
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import random
import time

class Trainer:
    """
    A class to train models on the WESAD dataset\\
    The dataset features should be first extracted and put it in the `X`
    """
    def __init__(self, models_params:dict[ClassifierMixin, dict[str, list]], X:pd.DataFrame, y:pd.DataFrame, split=True) -> None:
        """
        Parameters
        -----
        `model_params`: a dictionary of models\\
        Keys are the models, and values are the corresponding parameters
        `X`: Data Frame\\
        columns are the extracted features of windows\\
        rows are the values of each window
        `y`: Data Frame\\
        the label of each window
        """
        self.models:list[GridSearchCV] = []
        for model in models_params:
            self.models.append(GridSearchCV(model(), models_params[model], scoring='f1_macro', cv=5))
        if split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = X.copy(), None, y.copy(), None
        self._best_estimator = None
        self._f1_avg = 0.0
    
    def fit(self, verbose:int=0) -> None:
        """
        fit all the passed models and find the best estimator
        """
        # fit the models
        for model in self.models:
            if verbose:
                print(end=f'Training {model.estimator}')
                start_time = time.time()
            model.fit(self.X_train, self.y_train)
            if verbose:
                print(f"\rTraining {model.estimator} completed in {time.time() - start_time} secs")

            
            if model.best_score_ > self._f1_avg:
                self._f1_avg, self._best_estimator = model.best_score_, model.best_estimator_
        

    def predict_all(self, verbose:int=0, X_test=None) -> dict[ClassifierMixin, np.array]:
        if X_test == None:
            X_test = self.X_test
        preds = {}
        for model in self.models:
            if verbose:
                print(f'Predicting {model}')
            preds[model] = model.predict(X_test)
        
        self.predictions = preds
        return preds
    
    def report_all(self, verbose:int=0, X_test=None):
        if not hasattr(self, 'predictions'):
            self.predict_all(verbose=verbose, X_test=X_test)
        reports = {}
        for model in self.predictions:
            report = classification_report(y_true=self.y_test, y_pred=self.predictions[model], output_dict=True)
            reports[model] = report
        
        return reports
    

class WESADTrainer:
    def __init__(self, models_params:dict[ClassifierMixin, dict[str, list]], base_folder='WESAD/', filenames:list[str]=None) -> None:
        """
        Parameters
        ----
        `base_folder`: string\\
        the base location of datasets
        `filenames`: list of strings\\
        the list of filenames for each subject\\
        files must be in this location: <`base_folder`>/<subject_id>/data/<filename>\\
        the default is 
        """
        if filenames == None:
            filenames = ['chest_ACC', 'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_Resp', 'chest_Temp', 
                         'wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP']
        self.base_folder = base_folder
        self.filenames = filenames
        self.models_params = models_params
        self._best_estimator = {}
        self.X_train, self.y_train = {}, {}
        
        
    def predict_all(self, verbose=0):
        """
        Parameters:
        -----
        `test`: the test subject

        """
        # for testNo in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
        for testNo in [2, 3, 4]:
            test = f"S{testNo}"
            if verbose:print(f"Testing {test}")
            self.load_data(test=test, verbose=verbose) # self.sensors_dfs set to dictionary of DFs

            if verbose:
                print({sen: self.sensors_dfs[sen].shape for sen in self.sensors_dfs.keys()})
            self.fit(verbose=verbose) # apres, models are in self._best_estimator
            test_sensors = {sensor: [] for sensor in self.filenames}
            for filename in self.filenames:
                test_sensors[filename].append(pd.read_csv(self.base_folder + test + '/data/' + filename + '.csv'))
            test_sensors_dfs = {sensor: pd.concat(test_sensors[sensor]) for sensor in test_sensors}
            if verbose:
                print({sen: (test_sensors_dfs[sen].shape, test_sensors_dfs[sen].columns) for sen in test_sensors_dfs.keys()})
            report = self.report(test_sensors_dfs)
            self.X_test = test_sensors_dfs
            self.y_pred = self.prediction

            if verbose:print(f"Report of {test}: {report}")

    def load_data(self, test, verbose:int=0):
        sensors = {sensor: [] for sensor in self.filenames}
        
        for s in os.listdir(self.base_folder):
            if '.' in s and s != test: #subjects are in the folders
                continue
            for filename in self.filenames:
                sensors[filename].append(pd.read_csv(self.base_folder + s + '/data/' + filename + '.csv'))
            if verbose:print(end=f'\rSubject #{s} data loaded')
        if verbose:
            print()
        self.sensors_dfs = {sensor: pd.concat(sensors[sensor]) for sensor in sensors} # a dictionary, keys are sensors, values are the data for all the subjects
        if verbose:
            print("data frames concated successfully")

        
    def fit(self, verbose:int=0):
        if not hasattr(self, 'sensors_dfs'):
            raise RuntimeError("Before fitting data, first load it!")
        for sensor in self.sensors_dfs:
            df = self.sensors_dfs[sensor]
            X = df[[col for col in df.columns if col != 'label']]
            y = df['label']
            trainer = Trainer(self.models_params, X, y, split=False)
            trainer.fit(verbose=max(verbose-1, 0))
            self._best_estimator[sensor] = trainer._best_estimator
            self.X_train[sensor] = trainer.X_train
            self.y_train[sensor] = trainer.y_train
            if verbose: print(f'Sensor {sensor} done!')
            del trainer
        if verbose:print()

        
    def predict(self, X_test:dict[str, list], verbose:int=0):
        preds = {}
        final_pred = np.zeros((len(list(X_test.values())[0]),))
        for sensor in self._best_estimator:
            df = X_test[sensor]
            X = df[[col for col in df.columns if col != 'label']]
            y = df['label']
            pred = self._best_estimator[sensor].predict(X)
            preds[sensor] = pred
        
        for i in range(len(preds)):
            counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for sensor in self._best_estimator:
                counts[preds[sensor][i]] += 1
            popular = max(counts.items(), key=lambda x: x[1])[1] # the most vote number
            items = [x[0] for x in counts.items() if x[1] == popular] # those with the most vote
            final_pred[i] = random.sample(items, 1)[0] # choose one of them randomly
        self.prediction = final_pred
        self.y_test = y
        return final_pred
    
    def report(self, X_test:dict[str, list]=None, verbose:int=0):
        if not hasattr(self, 'prediction'):
            assert X_test != None
            self.predict(X_test=X_test, verbose=verbose)
        report:dict = classification_report(y_true=self.y_test, y_pred=self.prediction, output_dict=True)
        return report





            

        
        



        


