#import "templates/proposal.typ": project
#import "metadata.typ": details

#show: body => project(details, body)

#set heading(numbering: "1.1")
= Introduction

Medical imaging is essential for diagnosis and treatment planning, with magnetic resonance imaging (MRI) playing a crucial role.
In pneumonology, pulmonary MRI scans offer valuable insights @sodhi2021.
However, respiratory motion during imaging can introduce artefacts that complicate analysis.
Image registration aligns multiple scans for improved diagnostic accuracy.

While effective traditional registration techniques can be computationally expensive @fan2019, deep learning has demonstrated potential for faster and more accurate image analysis tasks @dirnet2017 @voxelmorph2019 @cyclemorph2020 @vitvnet2021 @transmorph2022.
This thesis explores the application of deep learning for the automatic registration of pulmonary MRI images.
The goal is to identify a deep learning framework that offers both accuracy comparable to traditional methods and the computational efficiency desired for faster clinical workflows.

= Problem

Medical imaging is essential in modern diagnostics.
Magnetic resonance imaging (MRI) is a commonly used technique to examine various medical conditions in patients.
One such field is pneumonology, where pulmonary images are used for diagnosis.
However, after a scan, MR images have respiratory motion, which is an issue for an accurate diagnosis @SyN2010.
Additionally, non-linear motions, non-stationary noise and intensity inhomogeneity increase the complexity of the problem @SyN2008. 
Therefore, one has to perform a registration to align an image series.
These registered images are necessary for physicians and other systems to perform any diagnostics @Voskrebenzev2017.

The process of aligning images is critical @SyN2008 @SyN2010.
Yet, classical techniques are computationally expensive @SyN2010 @dirnet2017, leading to time-consuming operations.
To overcome this challenge, researchers have suggested using deep learning techniques @dirnet2017 @voxelmorph2019 @cyclemorph2020 @vitvnet2021 @transmorph2022, which have proven effective in addressing the computational obstacles. 
Despite advancements in deep learning models, challenges persist because the models need to learn non-trivial relationships while dealing with temporal and spatial variations during the respiratory cycle and missing anatomical structural information in MR images.

As a result, traditional methods still provide comparable results to deep learning techniques in some cases.

= Motivation

Advancements in technology allow for modern diagnostics, enabling physicians to provide better, more reasonable diagnoses, which ultimately benefit patients in their treatment.
The paper @Voskrebenzev2017 presented a method called Phase-Resolved Functional Lung (PERFUL) MRI.
They demonstrated the feasibility of deriving the mapping of perfusion and ventilation in pulmonary images without radiation and contrasting agents, and patients can continue breathing normally.
Therefore, there is an interest in using it for clinical pulmonary tests.

The method relies on automatic processing, where image registration is the most time-consuming.
Minimising the time would lead to a faster output of diagnostic images and potentially better diagnosis response times.
Convolutional networks (CNN) @dirnet2017 @voxelmorph2019 @cyclemorph2020 and transformers @vitvnet2021 @transmorph2022 demonstrated similar performance to classical registration methods @SyN2008 due to their ability to learn complex patterns.
Of course, these deep-learning methods require training; however, they can register images in seconds or less in a single pass @dirnet2017 @voxelmorph2019 @cyclemorph2020 @vitvnet2021 @transmorph2022.

Automated registration is a crucial image processing step, which can be time-consuming @SyN2010. 
Recent studies in other medical fields by @dirnet2017 @voxelmorph2019 @cyclemorph2020 using convolutional networks (CNN) and @vitvnet2021 @transmorph2022 using hybrid-networks (CNN and transformer) have demonstrated outcomes comparable to traditional registration methods @SyN2008 @SyN2010.
The success of these deep-learning approaches is due to their ability to identify intricate patterns. 
Even using the central processing unit (CPU), they can register images in seconds with just one pass, and one can reduce the time even more by employing a graphics processing unit (GPU).

= Objective

The objective of the master thesis is to determine a deep-learning framework for automatic image registration.
The essential criterion is that the machine learning (ML) models show comparable performance regarding registration accuracy and error.
Not meeting the criterion would lead to exclusion in a clinical context.
Ideally, we identify an ML architecture that surpasses the traditional approach while offering the advantage of being computationally efficient, using a moderately complex model.

= Schedule

...
