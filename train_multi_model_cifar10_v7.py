from multi_task_adv_selective_utils import *
# seed_value= 321456798

# import os
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# # os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ['PYTHONHASHSEED']=str(seed_value)

# import random
# random.seed(seed_value)

# import numpy as np
# np.random.seed(seed_value)

# import tensorflow as tf
# # tf.random.set_seed(seed_value)
# # for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# from keras import backend as K
# # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# # K.set_session(sess)
# # for later versions:
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)

from models.cifar10.multi_task_adv_selective_model_cifar10_v7b import multi_cifar10_model as model_cifar10_adv_v7b
from models.cifar10.multi_task_adv_selective_model_cifar10_v7 import multi_cifar10_model as model_cifar10_adv_v7a
import sys
from keras import optimizers

mode = 'train'
no_defense_h5= sys.argv[1] # "cifar10_model_3.h5" #sys.argv[1]
filename_a = sys.argv[2] #"detector_cifar10_v7_x3.h5" #sys.argv[2]
filename_b = sys.argv[3] #"detector_cifar10_v7b_x3.h5" #sys.argv[3]
coverage = float(sys.argv[4]) #1.0
coverage_th = float(sys.argv[5]) #0.995
alpha = 0.5
normalize_mean = False

model_class_v7a = model_cifar10_adv_v7a(mode=mode, no_defense_h5=no_defense_h5, filename=filename_a, 
    coverage=coverage, coverage_th=coverage_th, alpha=alpha, normalize_mean=normalize_mean)

coverage = float(sys.argv[6]) #1.0
coverage_th = float(sys.argv[7]) #0.70
alpha = 0.5
normalize_mean = False
model_class_v7b = model_cifar10_adv_v7b(mode=mode, no_defense_h5=no_defense_h5, filename_a=filename_a, filename_b=filename_b, 
    coverage=coverage, coverage_th=coverage_th, alpha=alpha, normalize_mean=normalize_mean)