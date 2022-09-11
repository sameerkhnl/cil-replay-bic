import numpy as np
from ratio_merge import ratio_merge

def get_cl_margins(logits, y):
    margins = np.zeros(logits.shape[0])
    corr = y
    high = np.argsort(logits, 1)[:,-1]
    s_high = np.argsort(logits,1)[:,-2]
    
    corr = np.expand_dims(corr,1)
    high = np.expand_dims(high,1)
    s_high = np.expand_dims(s_high,1)
    
    c = np.take_along_axis(logits, corr, 1)
    high = np.take_along_axis(logits, high, 1)
    s_high = np.take_along_axis(logits, s_high, 1)
    
    pos = np.where(c == high)[0]
    neg = np.where(c != high)[0]
    
    margins[pos] = np.subtract(np.squeeze(c[pos]),np.squeeze(s_high[pos]))
    margins[neg] = np.subtract(np.squeeze(c[neg]),np.squeeze(high[neg]))
    
    return margins

def indexes_based_on_margin(margins, y, nb_per_class, highfirst=True):
    indexes = []
    for class_id in np.unique(y):
        class_indexes = np.where(y == class_id)[0]
        idxs = np.argsort(margins[class_indexes])
        idxs = np.flip(idxs) if highfirst else idxs
        idxs = idxs[:nb_per_class]
        indexes.append(class_indexes[idxs])
    return np.concatenate(indexes)
    

def get_correct_ids_sorted(x,y,t,args,margin='high_first', top_two_only = True):
    logits = args.logits
    dict_sorted = {}
    for class_id in np.unique(y):
        class_indexes = np.where(y == class_id)[0]
        y_hat = np.argmax(logits[class_indexes], axis=1)
        _class_indexes_correct = np.where(y_hat == class_id)
        class_indexes_correct = class_indexes[_class_indexes_correct]
        
        if top_two_only:
            top_two = np.sort(logits[class_indexes_correct], axis=1)[:,-2:]
            diff_top_two = np.diff(top_two, axis=1).flatten()
            argv = np.argsort(diff_top_two)
            # max_margin,min_margin,avg_margin = np.max(diff_top_two),np.min(diff_top_two),np.mean(diff_top_two)
            # print()
            # print(f'Class: {class_id}')
            # print(f'cl_margin (max, avg, min) : {max_margin:.3f}, {avg_margin:3f}, {min_margin:3f}')
            # print()
        else:
            _sorted = np.sort(logits[class_indexes_correct ], axis=1)
            _max = np.max(_sorted, axis=1)
            _max = np.expand_dims(_max,1)
            diff_max = np.sum(_max - _sorted, axis=1)
            argv = np.argsort(diff_max)
            
        if margin == 'high_first':
            argv = np.flip(argv)
            
        dict_sorted[class_id] = class_indexes_correct[argv]
        
    return dict_sorted   
    
def get_incorrect_ids(x,y,t,args):
    logits = args.logits
    dict_idx_incorrect = {}
    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        idx_incorrect = np.where(logits[class_indices].argmax(1) != y[class_indices])[0]
        np.random.shuffle(idx_incorrect)
        dict_idx_incorrect[class_id] = class_indices[idx_incorrect]
    return dict_idx_incorrect

def get_final_idx_from_dict(dict_merged, nb_per_class):
    idx = []
    for k,v in dict_merged.items():
        dict_merged[k] = dict_merged[k][:nb_per_class]
        idx.append(dict_merged[k])
        # print(f'no samples class{k}: {len(np.unique(dict_merged[k]))}')
    idx = np.concatenate(idx)
    return idx

def merge_ratio(dict1, dict2, ratio=1.0):
    dict_merged = {}
    for k,v in dict1.items():
        gen = ratio_merge(dict1[k], dict2[k], ratio)
        idx = np.array(list(gen))
        dict_merged[k] = idx
    return dict_merged

#Only include samples that were classified incorrectly
class Incorrect_Only:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        inc = get_incorrect_ids(x,y,t,args)
        idx = get_final_idx_from_dict(inc, nb_per_class)
        return x[idx],y[idx],t[idx]

class CL_MARGIN_HIGH:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        margins = get_cl_margins(logits=args.logits, y=y)
        indexes = indexes_based_on_margin(margins, y, nb_per_class, highfirst=True)
        # print(f'Average margin value using cl_margin_high: {np.mean(margins[indexes]):.3f}')
        return x[indexes], y[indexes], t[indexes]

#samples with low cl_margin; more difficult examples
class CL_MARGIN_LOW:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        margins = get_cl_margins(logits=args.logits, y=y)
        indexes = indexes_based_on_margin(margins, y, nb_per_class, highfirst=False)
        print(f'Average margin value using cl_margin_low: {np.mean(margins[indexes]):.3f}')

        return x[indexes], y[indexes], t[indexes]

#80% easy examples and 20% difficult examples
class CL_MARGIN_H80_L20:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        # dict_inc = get_incorrect_ids(x,y,t,args)
        margins = get_cl_margins(logits = args.logits, y=y)
        indexes_h = indexes_based_on_margin(margins, y, nb_per_class, highfirst=True)
        indexes_l = indexes_based_on_margin(margins, y, nb_per_class, highfirst=False)
        gen = ratio_merge(indexes_h, indexes_l, 0.5)
        indexes = np.array(list(gen))
        return x[indexes], y[indexes], t[indexes]

class CL_MARGIN_HIGH_ALL:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        dict_inc = get_incorrect_ids(x,y,t,args)
        dict_high = get_correct_ids_sorted(x,y,t,args, margin='high_first', top_two_only=False)
        ratio = 1 - args.incorrect_ratio
        dict_merged = merge_ratio(dict_high, dict_inc)
        idx = get_final_idx_from_dict(dict_merged, nb_per_class)
        return x[idx], y[idx], t[idx]

class CL_MARGIN_LOW_ALL:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        dict_inc = get_incorrect_ids(x,y,t,args)
        dict_low = get_correct_ids_sorted(x,y,t,args, margin='low_first', top_two_only=False)
        ratio = 1 - args.incorrect_ratio
        dict_merged = merge_ratio(dict_low, dict_inc)
        idx = get_final_idx_from_dict(dict_merged, nb_per_class)
        return x[idx], y[idx], t[idx]

class CL_MARGIN_CLASS_AVG:
    def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, args, nb_per_class) -> np.ndarray:
        margins = get_cl_margins(args.logits, y)
        indexes = []
        for class_id in np.unique(y):
            class_indexes = np.where(class_id == y)[0]
            m_avg = np.mean(margins[class_indexes])
            diff = np.abs(m_avg - margins[class_indexes]) 
            idxs = np.argsort(diff)[:nb_per_class]
            indexes.append(class_indexes[idxs])
        indexes = np.concatenate(indexes)
        return x[indexes], y[indexes], t[indexes]
   



