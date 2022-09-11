#model average at each task
from types import new_class
from Args import Args
import numpy as np
from continuum import Logger

#model confidence at different tasks
model_confidence= {}
 #class means scores for new classes of the current state.
init_class_means = {}
#class means scores for past classes of the current state.
current_class_means = {}
import math

def task_id_by_class(args:Args, class_id):
    return math.floor((class_id - args.initial_increment)/args.increment) + 1

def compute_model_conf(logits, y, new_classes, tid:int):
    assert tid > 0 , "model confidence should only be calculated when {} > 0".format(tid)
    _max = []

    for c in new_classes:
        class_indexes = np.where(y == c)[0]
        m = np.max(logits[class_indexes], 1)
        _max.append(m)
    model_confidence[tid] = np.mean(_max)

def add_init_class_means(logits, y, new_classes, tid:int):
    assert tid > 0 , "class means should only be calculated when {} > 0".format(tid)
    assert logits.shape[0] == y.shape[0], "unequal number of logits ({}) and targets ({})".format(logits.shape[0], y.shape[0])

    for c in np.unique(new_classes):
        class_indexes = np.where(y == c)[0]
        m = np.mean(logits[class_indexes][:,c])
        init_class_means[c] = m
    
#adds the mean scores on old classes in the current task
def add_current_class_means(logits, y, old_classes,tid:int):
    global current_class_means 
    current_class_means = {}
    assert tid > 0 , "later class means should only be calculated when {} > 1".format(tid)
    assert logits.shape[0] == y.shape[0], "unequal number of logits ({}) and targets ({})".format(logits.shape[0], y.shape[0])

    for o in np.unique(old_classes):
        class_indexes =  np.where(y == o)[0]
        m = np.mean(logits[class_indexes][:,o])
        current_class_means[o] = m

def _get_logits_rectified(logits_pred, old_classes, args, tid):
    import copy
    rectified_logits = copy.deepcopy(logits_pred)
    assert tid > 0 , "later class means should only be calculated when {} > 1".format(tid)

    to_rectify_ids = np.where(np.invert(np.in1d(np.argmax(logits_pred, 1), old_classes)))[0]

    for i in to_rectify_ids:
        for j in old_classes:
            r1 = init_class_means[j] / current_class_means[j]
            old_taskid = task_id_by_class(args, j)
            r2 = model_confidence[tid] / model_confidence[old_taskid]
            rect = r1*r2
            rectified_logits[i][j] *= rect
    return rectified_logits

def classify(logits_pred, y, t, old_classes, args:Args, tid):
    logits_rectified = _get_logits_rectified(logits_pred, old_classes, args, tid)
    preds = np.argmax(logits_rectified, 1)
    logger = Logger()
    logger.add([preds, y, t], subset='test')
    return logger




    













