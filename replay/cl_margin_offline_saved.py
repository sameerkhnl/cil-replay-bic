import pickle
from pathlib import Path
import numpy as np

def load_indexes_from_file(dataset):
    indexes = None
    pkl_filepath = Path.cwd() / 'exemplars_offline_cl_margin_high' / f'{dataset}.pkl'
    print(pkl_filepath)
    with open(pkl_filepath,'rb') as fp:
        indexes = pickle.load(fp)
    return indexes

class CL_Margin_High_Offline:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        dataset = args.dataset
        loaded = load_indexes_from_file(dataset)
        x_ = []
        y_ = []
        t_ = []

        for class_id in np.unique(y):
            x_class = loaded[class_id][:nb_per_class]
            x_.append(x_class)
            y_.append(np.full(len(x_class),class_id))
            t_.append(np.full(len(x_class),t[0]))
        x_ = np.concatenate(x_)
        y_ = np.concatenate(y_)
        t_ = np.concatenate(t_)
        return x_,y_,t_

class CL_Margin_Low_Offline:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        dataset = args.dataset
        loaded = load_indexes_from_file(dataset)
        x_ = []
        y_ = []
        t_ = []

        for class_id in np.unique(y):
            x_class = loaded[class_id][-nb_per_class:]
            x_.append(x_class)
            y_.append(np.full(len(x_class),class_id))
            t_.append(np.full(len(x_class),t[0]))
        x_ = np.concatenate(x_)
        y_ = np.concatenate(y_)
        t_ = np.concatenate(t_)
        return x_,y_,t_