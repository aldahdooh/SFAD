# About
## Authors

|ALDAHDOOH Ahmed, HAMIDOUCHE Wassim, DEFORGES Olivier|
|:-:|
|Univ Rennes, INSA Rennes, CNRS, IETR - UMR 6164, F-35000 Rennes, France|
|email: [ahmed.aldahdooh(at)insa-rennes(.)fr](mailto:ahmed.aldahdooh@insa-rennes.fr)|

## Paper Link
Please refer to [link](https://github.com/aldahdooh/SFAD).

## Citation
```

```

## Abstract
Security-sensitive applications that relay on Deep Neu-ral  Networks  (DNNs)  are  vulnerable  to  small  perturba-tions crafted to generate Adversarial Examples (AEs) thatare imperceptible to human and cause DNN to misclassifythem.   Many  defense  and  detection  techniques  have  beenproposed.   The  state-of-the-art  detection  techniques  havebeen designed for specific attacks or broken by others, needknowledge  about  the  attacks,  are  not  consistent,  increasemodel parameters overhead, are time-consuming, or havelatency in inference time. To trade off these factors, we pro-pose a novel unsupervised detection mechanism that usesthe  selective  prediction,  processing  model  layers  outputs,and  knowledge  transfer  concepts  in  a  multi-task  learningsetting. It is called **_Selective and Feature based AdversarialDetection (SFAD)_**.  Experimental results show that the pro-posed approach achieves comparable results to the state-of-the-art methods against tested attacks in white, black, andgray boxes scenarios. Moreover, results show that SFAD isfully robust against High Confidence Attackss (HCAs) forMNIST and partially robust for CIFAR-10 datasets.

# SFAD:Selective and Feature based AdversarialDetection
## _Architecture_
### Detector (High-level Model Archeticture)
The  input  sample  is passed  to  the  CNN  model  to  get  outputs  of N-last  layers  to  be processed in the detector classifiers.  SFAD yields prediction and selective probabilities to determine the prediction class of the inputsample and whether it is adversarial or not.
<p align="center">
  <img src="{{site.url}}/images/abstract_model.png" width="1200" title="High-level Model Archeticture">
</p>

### Detector (Model Archeticture)
<p align="center">
  <img src="{{site.url}}/images/detector_design.png" width="1200" title="Model Archeticture">
</p>

### Selective AEs Classifiers
<p align="center">
  <table>
      <tr>
        <td><img src="{{site.url}}/images/module_feature.png" width="1200" title="Selective AEs Classifiers"></td>
        <td>Textes</td>
      </tr>
   </table>
</p>

### Selective Knowledge Transfere Classifier

## _Performance at (FP=10%)_
### Performance against _white-box_ attacks
### Performance against _black-box_ attacks
### Performance against _grey-box_ attacks
