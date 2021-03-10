### Authors

<p align="center">
  <table>
      <tr>
        <td>Ahmed Aldahdooh</td>
        <td>Wassim Hamidouche</td>
        <td>Olivier Deforges</td>
      </tr>
      <tr>
        <td colspan="3">Univ Rennes, INSA Rennes, CNRS, IETR - UMR 6164, F-35000 Rennes, France</td>
      </tr>
      <tr>
        <td colspan="3"><a href = "mailto:ahmed.aldahdooh@insa-rennes.fr">ahmed.aldahdooh@insa-rennes.fr</a></td>
      </tr>
  </table>
</p>

### Paper
[Preprint](https://arxiv.org/abs/2103.05354)

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
  <table>
      <tr>
        <td><img src="{{site.url}}/images/detector_design.png" width="1200" title="Model Archeticture"></td>
      </tr>
      <tr>
        <td>It is believed that the lastN-layers in the DNN have potentials  in  detecting  and  rejecting  AEs.   At this very high level of presentation, AEs are indistinguishable from samples of the target class. Unlike other works, in this work, 1) the representative of the last layer outputs Z<sub>j</sub>, as features, are processed.  2) Multi-Task Learning (MTL) is used.   MTL has an advantage of combining related tasks with one or more loss function(s) and  it  does  better  generalization  especially  with  the  help of  the  auxiliary  functions. 3) selective prediction concept is utilized. In order to build safe DL models, prediction uncertainties,S<sub>qj</sub> and S<sub>t</sub>, have to be estimated and a rejection mechanism to control the uncertaintyhas to be identified as well.  Here, a Selective and Feature based Adversarial Detection (SFAD) method is demonstrated. As depicted in the Figure, SFAD consists of two main blocks (in grey); the selective AEs classifiers block and the selective knowledge transfer classifier block.   Besides the DNN prediction,P<sub>b</sub>, the two blocks give as output 1) detector  prediction  probabilities,P<sub>qj</sub>,  and P<sub>t</sub>,  and  selective probabilities, S<sub>qj</sub> and S<sub>t</sub>. The detection blocks (in red) take these probabilities to identify adversarial status of input x.</td>
      </tr>
   </table>
</p>

### Selective AEs Classifiers
<p align="center">
  <table>
      <tr>
        <td><img src="{{site.url}}/images/module_feature.png" width="1200" title="Selective AEs Classifiers"></td>
      </tr>
      <tr>
        <td>Prediction Task: we process the representative lastN-layer(s) outputs Z<sub>j</sub> with different ways  in  order  to  make  clean  input  features  more  unique. This will limit the feature space that the adversary uses to craft  the  AEs.   Each  of  lastN-layer  output  has its  own  feature  space  since  the  perturbations  propagation became clear when DNN model goes deeper.  That makes each of the <i>N</i> classifiers to be trained with different feature space.  Hence, combining and increasing the number of <i>N</i> will enhance the detection process. The aim of this block isto build <i>N</i> individual classifiers.  The Figure shows the architecture of one classifier. The input of each classifier is one or more of N-layers representative outputs. As  depicted  in  the Figure each  selective  classifier  consists  of  different  processing  blocks;  auto-encoders  block,up/down-sampling   block,   bottleneck   block,   and   noiseblock.  These blocks aim at giving distinguishable features for input samples to let the detector recognises the AEs efficiently. <hr/>
        Selective  Task: The  aim  of  this  task  is  to  train the  prediction  task  with  the  support  of  selective  prediction/rejection  as  shown  in  the Figure.  The input of the selective task is the last layer representative output of the prediction task q<sub>j</sub>.The selective task architecture is simple.  It consists of onedense layer with ReLU activation and batch normalization(BN) layers followed by special Lambda layer that divides the output of BN by 10. Then it followed by one output dense layer with sigmoid activation. <hr/> 
        Auxiliary task: In the MTL models,  auxiliary task mainly comes to help generalizing the prediction task. Most of the MTL models focus on one main task andother/auxiliary tasks must be related.  Our main task in the classifier is to train a selective prediction for the input features and in order to optimize this task, low-level featureshave to be accurate for the prediction and selective tasks and not to be overfitted to one of these tasks. Hence, the original prediction process is considered as an auxiliary task.</td>
      </tr>
   </table>
</p>

### Selective Knowledge Transfere Classifier
<p align="center">
  <table>
      <tr>
        <td><img src="{{site.url}}/images/module_transfer.png" width="1200" title="Selective Knowledge Transfere Classifier"></td>
      </tr>
      <tr>
        <td>The idea behind this block is that each set of inputs (the confidence values of <i>Y</i> classes of <i>j</i> classifier) is considered as a special feature of the clean input and combining different sets  of  the  these  features  makes  the  features  morerobust.  Hence, we transfer this knowledge of clean inputs to the classifier.  Besides, in the inference time, we believe that AE will generate different distribution of the confidence values and if it was able to fool one classifier (selective AEs classifier), it may not fool the others.<hr/>
          Prediction Task:  the output of the prediction task of the selective AEs classifier q<sub>j</sub>(z) is class  prediction  values P<sub>qj</sub>.   These  confidence  values  are concatenated  with N outputs  from  prediction  tasks  to  be as  input Q of  the  selective knowledge transfer block as illustrated in the Figure. Its classifier consists of one or more dense layer(s) and yields confidence values for <i>Y</i> classes P<sub>t</sub>. <hr/>
        Selective  Task:  the selectivetask is also integrated in the knowledge transfer classifier to selectively predict/reject AEs.<hr/> 
        Auxiliary task: Similar to selective AEs classifiers,knowledge transfer classifier has the auxiliary network network, <i>h</i>,  that is trained using the same prediction task asassigned to <i>t</i></td>
      </tr>
   </table>
</p>

## _Performance at (FP=10%)_
Tables show selective, confidence, and ensemble detection accuracies  of  SFAD  prototype  for  MNIST  and  CIFAR10 datasets.   It  also  shows  the  baseline  DNN  prediction  accuracy  for  the  AEs  in  “Baseline  DNN”  row  and  for  the not detected AEs in “prediction” row.  The “Total” row is the total accuracy of detected and truly classified/predicted samples.
### Performance against _white-box_ attacks
<p align="center">
  <table>
    <tr>
        <td>We tested the proposed model with different typesof  attacks.  Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and Carlini-Wagner (CW), DF (DeepFool), and High Confidence Attacks (HCA) attacks are tested.  </td>
      </tr>
      <tr>
        <td><img src="{{site.url}}/images/table1.png" width="1600" title="Performance against white-box attacks"></td>
      </tr>
   </table>
</p>

### Performance against _black-box_ attacks
<p align="center">
  <table>
      <tr>
        <td>For the black-box attacks, Threshold Attack (TA), Pixel Attack (PA), and Spatial Transformation  attack  (ST) attacks  are  used  in  the  testing process</td>
      </tr>
      <tr>
        <td><img src="{{site.url}}/images/table2.png" width="1600" title="Performance against black-box attacks"></td>
      </tr>
   </table>
</p>

### Performance against _grey-box_ attacks
<p align="center">
  <table>
      <tr>
        <td>Gray-box scenario assumes that we knew only the model training  data  and  the  output  of  the  DNN  model  and  we did  not  know  the  model  architecture.   Hence,  we  trained two  models  as  substitution  models  named  Model#2  andModel#3 for MNIST and CIFAR-10.  Then, white-box based AEs are generated using the substitution models.</td>
      </tr>
      <tr>
        <td><img src="{{site.url}}/images/table3.png" width="1600" title="Performance against grey-box attacks"></td>
      </tr>
   </table>
</p>

## Citation
```

```
