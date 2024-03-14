#import "templates/proposal.typ": project
#import "metadata.typ": details
#import "@preview/tablex:0.0.8": tablex, rowspanx, colspanx

#show: body => project(details, body)

#set heading(numbering: "1.1")

= Problem

Magnetic resonance (MR) images have non-linear respiratory motion, an issue for diagnostics @Voskrebenzev2017 @SyN2010.
Moreover, non-stationary image noise and image intensity inhomogeneity increase the complexity of the problem @SyN2008.
Consequently, the registration process has to be robust against these issues and align an image series with minimal error. 

In the past decade, researchers have suggested using deep learning techniques to address computational costs @dirnet2017 @voxelmorph2019 @cyclemorph2020 @vitvnet2021 @transmorph2022.
Besides the fast registration, deep learning models are required to approximate the respiratory motion well for an accurate non-linear deformation of the images with temporal and spatial dependencies.
Additionally, training is unsupervised as anatomical structural information is often not provided.
As a result, traditional methods still provide comparable results to deep learning techniques.

= Objective

The objective of the master thesis is to determine a deep-learning framework/architecture for automatic image registration.
The essential criterion is that the machine learning (ML) model show comparable performance regarding registration accuracy and error.
Not meeting the criteria would lead to exclusion in a clinical context.
Ideally, we identify an ML architecture that surpasses the traditional approach while offering the advantage of being computationally efficient, using a moderately complex model.

= Neural networks

For the upcoming network architectures, we assume that images are affinely aligned and have a single channel.
One can extend the architectures to handle multi-channel images.
The affine transformation allows the models to focus on the non-linear motion only, potentially helping with the registration task.

#show "Vxm": `VoxelMorph`
#show "Cycm": `CycleMorph`
#show "Vitnet": `ViT-V-Net`
#show "Tsm": `TransMorph`

== Convolutional networks

Vxm @voxelmorph2019 is the current state-of-the-art for convolutional neural networks.
It is backbone network agnostic, meaning it is not required to use the proposed UNet @unet2015 architecture.
Further, Vxm uses a spatial transformer network @stn2016 to warp images based on a predicted deformation vector field (DVF).
As the spatial transformer network is differentiable, we backpropagate the image dissimilarity.
The paper proposed using the mean square error or cross-correlation in the loss function and optionally adding the dice coefficient.
For regularisation, they penalised large vector magnitudes to encourage smooth displacement. 

The paper @cyclemorph2020 presented a framework Cycm to enforce a model to approximate a diffeomorphic mapping.
They thoroughly discussed the implications of non-invertible one-to-one and onto mappings, which can lead to degenerated deformations (e.g., folding).
Hence, one has to train Cycm with two networks following this idea: $F: X -> Y, R: Y -> X$ where $F tilde.eq R^(-1)$.
Cycm uses Vxm as a backbone architecture with a modified loss function.
The loss consists of the registration loss (local cross-correlation, l2-loss), cycle-consistency loss ($X -> hat(Y) -> tilde(X)$ and it must be $X tilde.eq tilde(X)$) and an identity loss (stationary regions should not be changed, identical image = no deformation).
Further, they demonstrated the extension to a multiscale registration with a global and local network, showing promising results for a robust approach for large-volume registration.
Note that the training of two networks causes increased training time @transmorph2022.

The last paper, @davoxelmorph2022, indicated the need for attention mechanisms in CNNs because these neural networks are biased towards capturing local information and not modelling long-range dependencies.
Registration of Vxm improved by introducing attention modules to the used backbone network.
The idea is to attend to spatial relationships.

== Generative adversarial networks

Some researchers experimented with GAN-based frameworks (`cGAN` and `cycGAN`) @reviewA.
Across these papers, results seemed promising, especially for cross-modality.
However, the review @reviewA questioned the validity of the output.

The paper @gan2019 presented an implementation using `cycGAN`, achieving cycle consistency like Cycm.
Although the results strongly indicate a flexible technique with exceptional generalisation capabilities, we need to consider the lack of comparison to other network types.

== Hybrids
 
`ViT-V-Net` @vitvnet2021 addressed the limitations of CNNs regarding long-range spatial relations.
Yet, using vision transformers is unfeasible due to large volumes, making computations expensive.
Therefore, they used CNNs to encode images, providing low-resolution, high-level features.
These features get a positional embedding, which is then passed to the transformer.
A VNet @vnet2016 decodes the output.

Tsm @transmorph2022 adopted a similar approach.
The significant difference is that they replaced the encoder with a "swin transformer".
Of course, this leads to high parameter counts (approximately 45M), beating current CNNs and other transformers.
Further, their results showed that hybrid architectures leveraging transformers outperformed CNNs consistently by a small margin.

== Comparison

#tablex(
  columns: (auto, 1fr, 1fr, 1fr),
  align: horizon,

  /* --- header --- */
  rowspanx(2)[*Property*], colspanx(3)[*Type*],
  (), [_CNN_], [_Transformer_], [_GAN_],
  /* -------------- */

  [_Parameters_], [19M], [46M], [11M],
  [_Modality_], [monomodality], [monomodality], [multimodality],
  [_Lung Registration (CT)_], [yes], [yes], [yes],
  [_Attention_], [optional], [yes], [no],
)

Convolutional networks are a well-established computer vision approach @reviewA @reviewB.
They are lightweight and achieve acceptable performance.
Transformers are novel @reviewA and potentially better than CNNs @transmorph2022.
The critical factor is attention, which helps these networks to focus on essential parts of an image and understand spatial relationships.
GANs are promising but not well established, and the authors @reviewA expressed their concerns regarding output validity.

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

= Structure

+ Introduction
  + Problem
  + Motivation
  + Objectives
  + Outline
+ Background
  + Deep learning
  + Convolutional networks
+ Related work
+ Methodology
+ Experiment
  + Baseline
    + Theory (Theoretical foundation of the baseline model)
    + Implementation (Key implementation details)
    + Result (Present baseline performance)
  + Change N
    + Theory (Explain the theoretical basis for the modification)
    + Implementation (How the change was implemented)
    + Result (Performance after the change)
+ Evaluation
  + Objectives
  + results
  + Findings
  + Discussion
  + Limitations
+ Summary
  + Status
    + Realized Goals
    + Open Goals
  + Conclusion
  + Future Work
