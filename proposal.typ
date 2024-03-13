#import "templates/proposal.typ": project
#import "metadata.typ": details

#show: body => project(details, body)

#set heading(numbering: "1.1")

= Introduction

Medical imaging is vital in diagnostics and treatments, especially magnetic resonance (MR) imaging due to non-ionising radiation.
Pulmonary MR scans offer valuable insights @sodhi2021.
However, respiratory motion introduces artefacts spatially and temporally.
Hence, one employs image registration to align multiple MR images for diagnostics.

Traditional registration techniques are computationally expensive @fan2019 @SyN2010 and are a time-consuming operation.
On the other hand, deep learning techniques demonstrated to be potentially faster and more accurate @dirnet2017 @voxelmorph2019 @cyclemorph2020 @vitvnet2021 @transmorph2022 than traditional state-of-the-art.

This thesis explores the application of deep learning for the automatic registration of pulmonary MR images.
The goal is to identify a deep learning framework that offers accuracies comparable to traditional methods, efficient computational processing for faster clinical workflows, and the ability to handle respiratory motion artefacts effectively during image registration.

= Problem

MR images have non-linear respiratory motion, an issue for diagnostics @SyN2010.
Moreover, non-stationary image noise and image intensity inhomogeneity increase the complexity of the problem @SyN2008.
Consequently, the registration process has to be robust against these issues and align an image series with minimal error. 

In the past decade, researchers have suggested using deep learning techniques to address computational costs @dirnet2017 @voxelmorph2019 @cyclemorph2020 @vitvnet2021 @transmorph2022.
Besides the fast registration, deep learning models are required to approximate the respiratory motion well for an accurate non-linear deformation of the images with temporal and spatial dependencies.
Additionally, training is unsupervised as anatomical structural information is often not provided.
As a result, traditional methods still provide comparable results to deep learning techniques.

= Motivation

Advancements in technology allow for modern diagnostics, enabling physicians to provide better, more reasonable diagnoses, ultimately benefiting patients in their treatment.
The paper @Voskrebenzev2017 presented a method called Phase-Resolved Functional Lung (PERFUL) MR imaging.
They demonstrated the feasibility of deriving the mapping of perfusion and ventilation in pulmonary images without radiation and contrasting agents, and patients can continue breathing normally.
Therefore, there is an interest in using it for clinical workflows.

The method relies on automatic processing, where image registration is the most time-consuming step.
Minimising the time would lead to a faster output of diagnostic images and potentially better diagnosis response times.
Recent studies in other medical fields, such as neuroimaging and cardiology, using convolutional networks (CNN) @voxelmorph2019 and hybrid networks (CNN and transformer) @transmorph2022 demonstrated comparable registration accuracy to traditional registration methods @SyN2008.
Moreover, they showed that these models can register images in seconds with just one pass using a central processing unit (CPU).
One can reduce the time further by employing a graphics processing unit (GPU).

= Objective

The objective of the master thesis is to determine a deep-learning framework for automatic image registration.
The essential criterion is that the machine learning (ML) models show comparable performance regarding registration accuracy and error.
Not meeting the criteria would lead to exclusion in a clinical context.
Ideally, we identify an ML architecture that surpasses the traditional approach while offering the advantage of being computationally efficient, using a moderately complex model.

= Schedule

*Foundation and Baseline -- 4 Weeks*

- Revise deep-learning foundations
- Select network architecture -- define criteria for selection
- Identify loss metrics -- highly dependent on the selected architecture
- Implement the network -- yields a baseline model

*Improvement -- 6 Weeks*

- Select a concept to extend the baseline
- Implement the concept -- analyse the change in performance, determine the effect
- Compare improvement to baseline and previous model

*Optimization -- 2 Weeks*

- Tune hyperparameters -- find the optimal setting for the best model 
- Parameter reduction
- Fine-tuning -- additional training with a lower learning rate

*Evaluation -- 4 Weeks*

- Generate prediction and compare it to the traditional method
- Qualitative analysis -- check differences in deformation
- Quantitative analysis -- DICE (segmentation) and Jacobian

= Neural Networks

*CNNs*

@voxelmorph2019

@cyclemorph2020

@davoxelmorph2022

*Hybrid*

@vitvnet2021

@transmorph2022

*GANs*

@gan2019

*Comparison*

#figure(
  caption: [@reviewA],
  table(
    columns: (auto, auto, auto, auto),
    inset: 12pt,
    align: horizon,
    [*Network*], [*CNN*], [*Transformer*], [*GAN*],
    [_Parameters (approx)_], [19M], [46M], [11M],
    [_Modality_], [monomodality], [monomodality (multimodality not shown)], [multimodality],
    [_Used for Lung Registration_], [yes], [yes], [yes],
    [_Attention_], [optional], [yes], [no],
  )
)

- CNNs are established @reviewA @reviewB
- Transformers are novel @reviewA but potentially better then CNNs @transmorph2022
- GANs are promising with results but not well established + concerns regarding output validity @reviewA
