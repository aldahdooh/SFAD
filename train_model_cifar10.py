import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from models.cifar10.model_cifar10 import cifar10_model as model_cifar10_adv
from multi_task_adv_selective_utils import *
from keras import optimizers
from keras.models import Model
from art.attacks import FastGradientMethod, CarliniLInfMethod, ProjectedGradientDescent, DeepFool
from art.classifiers import KerasClassifier
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

mode = sys.argv[1]#'train'
filename = sys.argv[2]#"cifar10_model_logits.h5"
normalize_mean = False
adv_mode = False

model_class = model_cifar10_adv(mode=mode, filename=filename, normalize_mean=normalize_mean)

if mode == 'load':
  model = model_class.model
  learning_rate = 0.02
  lr_decay = 1e-6
  
  sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
  model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
  
  model.summary()
  loss_test, class_head_acc = model.evaluate(model_class.x_test, model_class.y_test)
  print('Loss::{:4.4f} and Accuracy::{:4.2f}%  on test data'.format(loss_test, class_head_acc * 100))