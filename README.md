# About
Code for Selective and Features based Adversarial Detection _**(SFAD)**_ technique that is presented in the paper "Selective and Features based Adversarial Detection" by ALDAHDOOH Ahmed, HAMIDOUCHE Wassim, and DEFORGES Olivier at ------- 2021 -------
###### Note: this work uses SelectiveNet concept that is implemented [here.](https://github.com/anonygit32/SelectiveNet)

# Reqiuerment
- Python 3 (tested on Python 3.8.5)
- Keras (tested on Keras 2.3.1)

# Training
- Train the baseline classifier for MNIST, and/or CIFAR10. The model will be saved in _checkpoints_ folder.
    * mode: string: train or load
    * model_name: string: your model file name
 ```
 # to train CNN model for mnist MNIST. It will train the model in models/mnist/model_mnist.py
 run train_model_mnist.py "mode" "model_name.h5" 
 ```
- Train the SFAD detector. The detector classifiers will be saved in _checkpoints_ folder.
    * mode: string: train or load
    * model_name: string: your model file name
    * datectors_a: string: your selecive adversarial examples classifiers file names. In checkpoints folder, three classifiers will be generated with names _**datectors_a_model_1.h5**_, _**datectors_a_model_2.h5**_, and _**datectors_a_model_3.h5**_.
    * detector_b: string: your selective knowledge transfer classifier file name. In checkpoints folder, the classifier will saved there.
    * overage_a coverage_a_th coverage_b coverage_b_th: float: SelectiveNet parameters.
 ```
 # to train a detector for MNIST. It will train the detectors implemented in models/mnist/multi_task_adv_selective_model_mnist_v7.py  and models/mnist/multi_task_adv_selective_model_mnist_v7b.py
 run train_multi_model_mnist_v7.py "mode" "model_name.h5" "datectors_a.h5" "detector_b.h5" coverage_a coverage_a_th coverage_b coverage_b_th
 ```
 # Testing
 - Generate the adversarial examples for the baseline classifier (for instance), model_name_fgsm_0.05.bytes (where 0.05 is the epsilon).
   * cahnge the adv_data_name array variables,inside the test_defense_mnist_model_v7.py, to include your adversarial examples paths.
   * control the ```th_indx``` to calculate thresholds that match your target rejection rate.
 - run 
 ```
 test_defense_mnist_model_v7.py "model_name.h5" "datectors_a.h5" "detector_b.h5"
 ```
