import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multi_task_adv_selective_utils import *
# import keras
import sys
from models.mnist.model_mnist import mnist_model as mnist_model_v0_class
from models.mnist.multi_task_adv_selective_model_mnist_v7b import multi_mnist_model as mnist_model_v7b_class
from keras import optimizers
from keras.models import Model
import xlsxwriter

#load model and test it 
mode = 'load'
filename_v0 = sys.argv[1]#"mnist_model_9.h5"
filename_v7a = sys.argv[2]#"detector_mnist_v7_x9_1.h5"
filename_v7b = sys.argv[3]#"detector_mnist_v7b_x9_1.h5"
xls_file = '/home/aaldahdo/adv_dnn/stats/stats_' + filename_v7a[:-3] +'.xlsx' #'/home/aaldahdo/adv_dnn/stats/stats_detector_mnist_v7_x1_1.xlsx' #'/home/aaldahdo/adv_dnn/stats/stats_' + sys.argv[2][:-3] +'.xlsx'
workbook = xlsxwriter.Workbook(xls_file, {'nan_inf_to_errors': True})

coverage = 0.8
alpha = 0.5
normalize_mean = False

model_class_v0 = mnist_model_v0_class(mode=mode, filename=filename_v0, normalize_mean=normalize_mean)
model_class_v7b = mnist_model_v7b_class(mode=mode, no_defense_h5=filename_v0, filename_a=filename_v7a, filename_b=filename_v7b, coverage=coverage, alpha=alpha, normalize_mean=normalize_mean)
model_class_v7a = model_class_v7b.model_class

model_v0 = model_class_v0.model
model_v7 = model_class_v7b.model
model_v7_model_1 = model_class_v7a.model_1
model_v7_model_2 = model_class_v7a.model_2
model_v7_model_3 = model_class_v7a.model_3
c = model_class_v7b.coverage
lamda = model_class_v7b.lamda
learning_rate = 0.01
lr_decay = 1e-6

def selective_loss(y_true, y_pred):
    loss = K.categorical_crossentropy(
        K.repeat_elements(
            y_pred[:, -1:], model_class_v7b.num_classes, axis=1) * y_true[:, :-1],
        y_pred[:, :-1]) + lamda * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
    return loss

def selective_acc(y_true, y_pred):
    g = K.cast(K.greater(y_pred[:, -1], 0.995), K.floatx())
    temp1 = K.sum(
        (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
    temp1 = temp1 / K.sum(g)
    return K.cast(temp1, K.floatx())

def coverage(y_true, y_pred):
    g = K.cast(K.greater(y_pred[:, -1], 0.995), K.floatx())
    return K.mean(g)

sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model_v0.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
model_v7.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                    loss_weights=[alpha, 1 - alpha],
                    optimizer=sgd, metrics=['accuracy', selective_acc])
model_v7_model_1.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                    loss_weights=[alpha, 1 - alpha],
                    optimizer=sgd, metrics=['accuracy', selective_acc])
model_v7_model_2.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                    loss_weights=[alpha, 1 - alpha],
                    optimizer=sgd, metrics=['accuracy', selective_acc])
model_v7_model_3.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                    loss_weights=[alpha, 1 - alpha],
                    optimizer=sgd, metrics=['accuracy', selective_acc])


model_v0.summary()
model_v7_model_1.summary()
model_v7_model_2.summary()
model_v7_model_3.summary()
model_v7.summary()
# loss_test, class_head_acc = model_v0.evaluate(model_class_v0.x_test, model_class_v0.y_test)
# print('Loss::{:4.4f} and Accuracy::{:4.2f}%  on test data'.format(loss_test, class_head_acc * 100))

# loss_test, _, _, _, _, selective_head_acc, aux_head_acc, _, rej_head_acc, _ = model_v7.evaluate(model_class_v7b.x_test, [model_class_v7b.y_test, model_class_v7b.y_test[:,:-1], model_class_v7b.y_test])
# print('Loss::{:4.4f} and Accuracy::{:4.2f}%  on test data'.format(loss_test, rej_head_acc * 100))

##########################################################################################
#
#                         MODEL v0 -- NO defence
#
##########################################################################################

predections_v0 = model_v0.predict(model_class_v0.x_test)
pred = np.reshape(np.argmax(predections_v0, axis=1), (10000,1))
indices_true = [i for i,v in enumerate(pred) if pred[i]==model_class_v0.y_test_labels[i]]
nSamples_true = len(indices_true)
x_test_true = model_class_v0.x_test[indices_true]
y_test_labels_true = model_class_v0.y_test_labels[indices_true]
y_test_true_v0 = model_class_v0.y_test[indices_true]

# adv_data_name = ['mnist_model_fgsm_0.05.bytes',
#                 'mnist_model_fgsm_0.075.bytes',
#                 'mnist_model_fgsm_0.1.bytes',
#                 'mnist_model_fgsm_0.2.bytes',
#                 'mnist_model_fgsm_0.4.bytes',
#                 'mnist_model_pgd_0.05.bytes',
#                 'mnist_model_pgd_0.075.bytes',
#                 'mnist_model_pgd_0.1.bytes',
#                 'mnist_model_pgd_0.2.bytes',
#                 'mnist_model_pgd_0.4.bytes',
#                 'mnist_model_df.bytes',
#                 'mnist_model_cw.bytes',
#                 'mnist_model_hc_0.05.bytes',
#                 'mnist_model_hc_0.075.bytes',
#                 'mnist_model_hc_0.1.bytes',
#                 'mnist_model_hc_0.2.bytes',
#                 'mnist_model_hc_0.4.bytes']
adv_data_name = ['_fgsm_0.05.bytes',
                '_fgsm_0.075.bytes',
                '_fgsm_0.1.bytes',
                '_fgsm_0.2.bytes',
                '_fgsm_0.3.bytes',
                '_fgsm_0.4.bytes',
                '_fgsm_0.6.bytes',
                '_fgsm_0.8.bytes',
                '_fgsm_1.bytes',
                '_pgd_0.05.bytes',
                '_pgd_0.075.bytes',
                '_pgd_0.1.bytes',
                '_pgd_0.2.bytes',
                '_pgd_0.3.bytes',
                '_pgd_0.4.bytes',
                '_pgd_0.6.bytes',
                '_pgd_0.8.bytes',
                '_pgd_1.bytes',
                '_df.bytes',
                '_cw.bytes',
                '_hc_0.4.bytes',
                '_th.bytes',
                '_p.bytes',
                '_sp.bytes'
                ]

flag_foreclean = True
for adv_name in adv_data_name:
    #load adv data
    open_file = open('adv_data/mnist/' + filename_v0[:-3] + adv_name , 'rb')
    # open_file = open('adv_data/mnist/mnist_transfer_2' + adv_name , 'rb')
    adv_bytes = open_file.read()
    attack_data = np.frombuffer(adv_bytes, dtype=np.float32)
    attack_data = attack_data.reshape(10000,28,28,1)
    open_file.close()

    #########################################################################################
    #                            Prepare inputs
    #########################################################################################


    intermediate_layer_model = Model(inputs=model_v0.input, outputs=model_v0.get_layer('l_16').output)
    l_1 = intermediate_layer_model.predict(attack_data)
    if len(l_1.shape)==2:
        l_1 = l_1.reshape(l_1.shape[0],1, 1, l_1.shape[1])
    elif len(l_1.shape)==4:
        l_1 = l_1.reshape(l_1.shape[0],l_1.shape[1], l_1.shape[2], l_1.shape[3])
    intermediate_layer_model = Model(inputs=model_v0.input, outputs=model_v0.get_layer('l_13').output)
    l_2 = intermediate_layer_model.predict(attack_data)
    if len(l_2.shape)==2:
        l_2 = l_2.reshape(l_2.shape[0],1, 1, l_2.shape[1])
    elif len(l_2.shape)==4:
        l_2 = l_2.reshape(l_2.shape[0],l_2.shape[1], l_2.shape[2], l_2.shape[3])
    intermediate_layer_model = Model(inputs=model_v0.input, outputs=model_v0.get_layer('l_10').output)
    l_3 = intermediate_layer_model.predict(attack_data)
    if len(l_3.shape)==2:
        l_3 = l_3.reshape(l_3.shape[0],1, 1, l_3.shape[1])
    elif len(l_1.shape)==4:
        l_3 = l_3.reshape(l_3.shape[0],l_3.shape[1], l_3.shape[2], l_3.shape[3])
    model_v7_model_1_adv = model_v7_model_1.predict(l_1)
    model_v7_model_2_adv = model_v7_model_2.predict(l_2)
    model_v7_model_3_adv = model_v7_model_3.predict(l_3)
    x_adv = np.concatenate((model_v7_model_1_adv[0][indices_true, :-1], model_v7_model_2_adv[0][indices_true, :-1], model_v7_model_3_adv[0][indices_true, :-1]), axis=1)


    ##########################################################################################
    #
    #                         MODEL v1 -- 3 heads (selection, lux, rejection)
    #
    ##########################################################################################
    if flag_foreclean:
        #generate adv_data from model v3a
        model_v7_model_1_clean = model_v7_model_1.predict(model_class_v7a.l_1_test)
        model_v7_model_2_clean = model_v7_model_2.predict(model_class_v7a.l_2_test)
        model_v7_model_3_clean = model_v7_model_3.predict(model_class_v7a.l_3_test)
        #----------------
        model_v7_model_1_clean_max = np.max(model_v7_model_1_clean[0][:,:-1], axis=1)
        model_v7_model_1_clean_max_sorted = np.sort(model_v7_model_1_clean_max)
        model_v7_model_2_clean_max = np.max(model_v7_model_2_clean[0][:,:-1], axis=1)
        model_v7_model_2_clean_max_sorted = np.sort(model_v7_model_2_clean_max)
        model_v7_model_3_clean_max = np.max(model_v7_model_3_clean[0][:,:-1], axis=1)
        model_v7_model_3_clean_max_sorted = np.sort(model_v7_model_3_clean_max)

        predections_v7_model_1_clean_sel_max = model_v7_model_1_clean[0][:,-1]
        predections_v7_model_1_clean_sel_max_sorted = np.sort(predections_v7_model_1_clean_sel_max)
        predections_v7_model_2_clean_sel_max = model_v7_model_2_clean[0][:,-1]
        predections_v7_model_2_clean_sel_max_sorted = np.sort(predections_v7_model_2_clean_sel_max)
        predections_v7_model_3_clean_sel_max = model_v7_model_3_clean[0][:,-1]
        predections_v7_model_3_clean_sel_max_sorted = np.sort(predections_v7_model_3_clean_sel_max)
        #----------------------

        
        x_clean = np.concatenate((model_v7_model_1_clean[0][indices_true, :-1], model_v7_model_2_clean[0][indices_true, :-1], model_v7_model_3_clean[0][indices_true, :-1]), axis=1)
        predections_v7_clean = model_v7.predict(x_clean)
        predections_v7_clean_detector = predections_v7_clean[0][:,:-1]
        class_clean_detector = np.reshape(np.argmax(predections_v7_clean_detector, axis=1), (x_clean.shape[0],1))
        clean_data_true = model_class_v7a.x_test[indices_true]
        predections_v0_clean_base = model_v0.predict(clean_data_true)
        class_clean_base = np.reshape(np.argmax(predections_v0_clean_base, axis=1), (clean_data_true.shape[0],1))
        indices_not_sim = [i for i,v in enumerate(class_clean_detector) if class_clean_base[i]!=class_clean_detector[i]]

        flag_use_average = False
        if flag_use_average:
            predections_v7_clean = [np.average([model_v7_model_1_clean[0][indices_true, :], model_v7_model_2_clean[0][indices_true, :], model_v7_model_3_clean[0][indices_true, :]], axis=0),
                                    np.average([model_v7_model_1_clean[1][indices_true, :], model_v7_model_2_clean[1][indices_true, :], model_v7_model_3_clean[1][indices_true, :]], axis=0)]
        predections_v7_clean_max = np.max(predections_v7_clean[0][:,:-1], axis=1)
        predections_v7_clean_max_sorted = np.sort(predections_v7_clean_max)
        predections_v7_clean_sel_max = predections_v7_clean[0][:,-1]
        predections_v7_clean_sel_max_sorted = np.sort(predections_v7_clean_sel_max)

        threshold_indx = []
        threshold = []
        threshold_s = []

        #----
        threshold_f1 = []
        threshold_f2 = []
        threshold_f3 = []
        #-----

        thre_per = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20 , 25, 30, 40, 50, 60, 60, 80, 90, 100]
        for i in thre_per:
            threshold_indx.append(np.ceil(len(predections_v7_clean_max_sorted) * (i/100.0)).astype(int))
        for indx in threshold_indx:
            threshold.append(predections_v7_clean_max_sorted[indx-1])
            threshold_s.append(predections_v7_clean_sel_max_sorted[indx-1])
            #-----
            threshold_f1.append(model_v7_model_1_clean_max_sorted[indx-1])
            threshold_f2.append(model_v7_model_2_clean_max_sorted[indx-1])
            threshold_f3.append(model_v7_model_3_clean_max_sorted[indx-1])
            #-----

        th_indx = 4
        th_selective = np.percentile(predections_v7_clean[0][:, -1], th_indx+2)
        th_selective1 = np.percentile(model_v7_model_1_clean[0][:, -1], th_indx)
        th_selective2 = np.percentile(model_v7_model_2_clean[0][:, -1], th_indx)
        th_selective3 = np.percentile(model_v7_model_3_clean[0][:, -1], th_indx)
        flag_not_max = True
        if flag_not_max:
            select1 = model_v7_model_1_clean[0][indices_true, -1] >= th_selective1
            select2 = model_v7_model_2_clean[0][indices_true, -1] >= th_selective2
            select3 = model_v7_model_3_clean[0][indices_true, -1] >= th_selective3
            select = predections_v7_clean[0][:, -1] >= th_selective
            select = np.logical_and(np.logical_and(select, select1), np.logical_and(select2, select3))
        else:
            th_selective = np.percentile([th_selective, th_selective1, th_selective2, th_selective3], 50)
            select1 = model_v7_model_1_clean[0][indices_true, -1] >= th_selective
            select2 = model_v7_model_2_clean[0][indices_true, -1] >= th_selective
            select3 = model_v7_model_3_clean[0][indices_true, -1] >= th_selective
            select = predections_v7_clean[0][:, -1] >= th_selective
            select = select &  select1 &  select2 &  select3

        resultx = {'th_selective': th_selective, 'th_selective1': th_selective1,
                   'th_selective2': th_selective2,'th_selective3': th_selective3, 
                   'threshold': threshold, 'threshold_f1': threshold_f1,
                   'threshold_f2': threshold_f2,'threshold_f3': threshold_f3, 
                   'threshold_rates': thre_per}

        clean_stat = []
        print('Information for clean data -----------------------------------')
        print('All accuracy    Selective Rej.  Feature Rej.    Feature Rej.(Mutual)     NotSim Rejection Accuracy    True Class.')
        count = 0
        for th in threshold:
            indices_select = [i for i,v in enumerate(select) if select[i]==False]
            th_f = np.max([th, threshold_f1[count], threshold_f2[count], threshold_f3[count]])
            indices_reject = [i for i,v in enumerate(predections_v7_clean[0]) if np.max(predections_v7_clean[0][i,:-1])<=th]
            indices_reject = [i for i,v in enumerate(predections_v7_clean[0]) if np.max(predections_v7_clean[0][i,:-1])<=th_f]
            reject_all = np.unique(np.concatenate((indices_not_sim, indices_select, indices_reject), axis=0))
            nSamples_rejected = len(reject_all)
            reject_rate = 100*nSamples_rejected/nSamples_true
            #-------------------------------------------
            pred_adv = np.reshape(np.argmax(predections_v7_clean[0][:,:-1], axis=1), (nSamples_true,1))
            pred_adv_true = [i for i,v in enumerate(pred_adv) if pred_adv[i]==y_test_labels_true[i]]
            pred_adv_true = np.reshape(pred_adv_true, (len(pred_adv_true), 1))
            true_in_rejected = np.sum(np.in1d(pred_adv_true, reject_all))
            true_net = len(pred_adv_true) - true_in_rejected
            acc = 100-reject_rate #100*true_net/nSamples_true
            acc_fail = 100*(nSamples_true-(nSamples_rejected+true_net))/nSamples_true
            FPR1 = true_in_rejected/nSamples_rejected

            feat_acc = 100*len(indices_reject)/nSamples_true
            feat_acc_no = 100*(nSamples_rejected-len(indices_select))/nSamples_true
            selec_acc = 100*len(indices_select)/nSamples_true
            not_sim_acc = 100*len(indices_not_sim)/nSamples_true
            print('({}) {:4.2f}\t{:4.2f}\t\t{:4.2f}\t\t{:4.2f}\t\t{:4.2f}\t\t\t{:4.2f}'.format(thre_per[count], acc+reject_rate, selec_acc, feat_acc_no, feat_acc, not_sim_acc, acc))
            
            clean_stat.append([thre_per[count], nSamples_true, len(indices_select), len(indices_reject), nSamples_rejected, true_net, true_in_rejected, 100*FPR1, 
                                acc+reject_rate, selec_acc, feat_acc, not_sim_acc, acc])
            count += 1
    
        worksheet1 = workbook.add_worksheet(name='clean')
        worksheet1.write_row(0, 0, ['Rejection rate',	'True Classified in Clean Data',
                            	'Selection reject',	'Feature reject',	'All reject',
                                'True Classification', 	'True in Reject',	'False positive rate',	
                                'All Accuracy', 'Selective Rejection Accuracy', 'Feature Rejection Accuracy' , 'NotSim Rejection Accuracy', 'True Accuracy'])
        flag_foreclean = False

        for row_num, data in enumerate(clean_stat):
            worksheet1.write_row(row_num+1, 0, data)
        
    worksheet2 = workbook.add_worksheet(name=adv_name[1:-6]+'_attacked')
    worksheet2.write_row(0, 0, ['Rejection rate',	'True Classified in Clean Data',
                            	'Selection reject',	'Feature reject',	'All reject',
                                'True Classification', 	'True in Reject',	'False positive rate',	
                                'All Accuracy B', 'Selective Rejection Accuracy', 'Feature Rejection Accuracy', 'NotSim Rejection Accuracy' ,'True Accuracy B' ,	'All Accuracy',	'True Accuracy' ])

    #------------------------------------------------------------------------------
    predections_v7_adv = model_v7.predict(x_adv) 
    predections_v7_adv_detector = predections_v7_adv[0][:,:-1]
    class_adv_detector = np.reshape(np.argmax(predections_v7_adv_detector, axis=1), (x_adv.shape[0],1))
    attack_data_true = attack_data[indices_true]
    predections_v0_adv_base = model_v0.predict(attack_data_true)
    class_adv_base = np.reshape(np.argmax(predections_v0_adv_base, axis=1), (attack_data_true.shape[0],1))
    indices_not_sim = [i for i,v in enumerate(class_adv_detector) if class_adv_base[i]!=class_adv_detector[i]]

    if flag_use_average:
            predections_v7_adv = [np.average([model_v7_model_1_adv[0][indices_true, :], model_v7_model_2_adv[0][indices_true, :], model_v7_model_3_adv[0][indices_true, :]], axis=0),
                                  np.average([model_v7_model_1_adv[1][indices_true, :], model_v7_model_2_adv[1][indices_true, :], model_v7_model_3_adv[1][indices_true, :]], axis=0)]

    if flag_not_max:
        select1 = model_v7_model_1_adv[0][indices_true, -1] >= np.percentile(model_v7_model_1_clean[0][:, -1], th_indx)
        select2 = model_v7_model_2_adv[0][indices_true, -1] >= np.percentile(model_v7_model_2_clean[0][:, -1], th_indx)
        select3 = model_v7_model_3_adv[0][indices_true, -1] >= np.percentile(model_v7_model_3_clean[0][:, -1], th_indx)
        select = predections_v7_adv[0][:, -1] >= th_selective
        select = np.logical_and(np.logical_and(select, select1), np.logical_and(select2, select3))
    else:
        select1 = model_v7_model_1_adv[0][indices_true, -1] >= th_selective
        select2 = model_v7_model_2_adv[0][indices_true, -1] >= th_selective
        select3 = model_v7_model_3_adv[0][indices_true, -1] >= th_selective
        select = predections_v7_adv[0][:, -1] >= th_selective
        select = select &  select1 &  select2 &  select3

    adv_stat = []
    print('----------------------------------- Information for adv data:' + adv_name + ' -----------------------------------')
    print('All accuracy    Selective Rej.  Feature Rej.    Feature Rej.(Mutual)     Not Simlar    True Class B.      True Class')
    count = 0
    for th in threshold:
        indices_select = [i for i,v in enumerate(select) if select[i]==False]
        th_f = np.max([th, threshold_f1[count], threshold_f2[count], threshold_f3[count]])
        indices_reject = [i for i,v in enumerate(predections_v7_adv[0]) if np.max(predections_v7_adv[0][i,:-1])<=th]
        indices_reject = [i for i,v in enumerate(predections_v7_adv[0]) if np.max(predections_v7_adv[0][i,:-1])<=th_f]
        reject_all = np.unique(np.concatenate((indices_not_sim, indices_select, indices_reject), axis=0))
        nSamples_rejected = len(reject_all)
        reject_rate = 100*nSamples_rejected/nSamples_true
        #-------------------------------------------
        pred_adv = np.reshape(np.argmax(predections_v7_adv[0][:,:-1], axis=1), (nSamples_true,1))
        pred_adv_true = [i for i,v in enumerate(pred_adv) if pred_adv[i]==y_test_labels_true[i]]
        pred_adv_true = np.reshape(pred_adv_true, (len(pred_adv_true), 1))
        true_in_rejected = np.sum(np.in1d(pred_adv_true, reject_all))
        true_net = len(pred_adv_true) - true_in_rejected
        acc = 100*true_net/nSamples_true
        acc_fail = 100*(nSamples_true-(nSamples_rejected+true_net))/nSamples_true
        FPR2 = true_in_rejected/nSamples_rejected

        feat_acc = 100*len(indices_reject)/nSamples_true
        feat_acc_no = 100*(nSamples_rejected-len(indices_select))/nSamples_true
        selec_acc = 100*len(indices_select)/nSamples_true
        not_sim_acc = 100*len(indices_not_sim)/nSamples_true
        #-----------
        not_rej = list(set(range(0, nSamples_true))-set(reject_all))
        if len(not_rej)!=0:
            attack_data_true = attack_data[indices_true]
            data_x = attack_data_true[not_rej]
            data_y = y_test_true_v0[not_rej]
            lo, ac = model_v0.evaluate(data_x, data_y, verbose=0)
            acc_base = 100*np.floor(ac*len(not_rej))/nSamples_true
        else:
            acc_base = 0
        #-----------
        print('({}) {:4.2f}\t{:4.2f}\t\t{:4.2f}\t\t{:4.2f}\t\t\t{:4.2f}\t\t{:4.2f}\t\t{:4.2f}'.format(thre_per[count], acc_base+reject_rate, selec_acc, feat_acc_no, feat_acc, not_sim_acc, acc_base, acc))

        adv_stat.append([thre_per[count], nSamples_true, len(indices_select), len(indices_reject), nSamples_rejected, true_net, true_in_rejected, 100*FPR2, 
                                acc_base+reject_rate, selec_acc, feat_acc, not_sim_acc, acc_base, acc, acc+reject_rate])
        count += 1

    for row_num, data in enumerate(adv_stat):
        worksheet2.write_row(row_num+1, 0, data)

workbook.close()
print('Done')