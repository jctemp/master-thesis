#import "templates/thesis.typ": project
#import "metadata.typ": details

#show: body => project(details, body)

= Introduction <ch-introduction>
// As aforementioned, the thesis tackles the problem of registration for a lung image series.
// The currently used tool is the advanced normalisation tool @avantsAdvancedNormalizationTools.
// However, it has the drawback of being significantly slower than deep learning alternatives @fuDeepLearningMedical2019 @chenTransMorphTransformerUnsupervised2021.
// Medicine is a continuously developing area that benefits from advancements in computer science.
// More computational capacity, enabling more complex and expensive operations, allowed the integration of computer-aided technologies in everyday clinical practice.
// Furthermore, the rapid development in machine learning creates new opportunities for researchers to introduce novel or optimise current state-of-the-art methods to increase efficiency in any aspect of computer-aided medicine.

// In the past decades, computer vision has become increasingly sophisticated due to the renaissance of convolutional neural networks (AlexNet, ResNet and Inception) and their achievements, replacing conventional state-of-the-art methods.
// In addition to image classification, convolutional neural networks are used for more advanced use cases like object detection, semantic segmentation, and image registration.
// Besides the change in computer vision, the transformer architecture, introduced in the paper "Attention is all need", became central to natural language processing (NLP).
// A slight modification reintroduced the transformer concept to computer vision with the name vision transformer, surpassing many convolutional neural networks.

// Computer vision is essential in medical imaging because object detection, semantic segmentation, and image registration are often required.
// Magnetic resonance (MR) imaging is a standard method used in clinical practice and is significant for many disciplines @reviewA because MR images are well-suited to examine various anatomical features.
// In addition, it does not require ionising radiation, which is a factor to consider if examining a patient.

// For this master's thesis, we will work with a novel method called phase-resolved functional lung (PERFUL) MR imaging @Voskrebenzev2017.
// It utilises MR lung images to compute regional perfusion and ventilation of the lung whilst not requiring radioactive tracers or patients to hold their breath.

== Problem
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Describe the problem that you like to address in your thesis to show the importance of your work. Focus on the negative symptoms of the currently available solution.
]

== Motivation
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Motivate scientifically why solving this problem is necessary. What kind of benefits do we have by solving the problem?
]

== Objectives and requirements
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Describe the research goals and/or research questions and how you address them by summarizing what you want to achieve in your thesis, e.g. developing a system and then evaluating it.
]

== Outline
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Describe the outline of your thesis
]

#include "chapters/background.typ"

= Related Work <ch-related-work>

#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Describe related work regarding your topic and emphasize your (scientific) contribution in contrast to existing approaches / concepts / workflows. Related work is usually current research by others and you defend yourself against the statement: “Why is your thesis relevant? The problem was al- ready solved by XYZ.” If you have multiple related works, use subsections to separate them.
]

= Methodology

= Experiment

== Baseline

=== Theory

=== Implementation 

=== Result

== Change 0

=== Theory

=== Implementation 

=== Result

== Change N

=== Theory

=== Implementation 

=== Result

= Evaluation

== Objectives
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Derive concrete objectives / hypotheses for this evaluation from the general ones in the introduction.
]

== Results
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Summarize the most interesting results of your evaluation (without interpretation). Additional results can be put into the appendix.
]

== Findings
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Interpret the results and conclude interesting findings
]

== Discussion
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Discuss the findings in more detail and also review possible disadvantages that you found
]

== Limitations
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Describe limitations and threats to validity of your evaluation, e.g. reliability, generalizability, selection bias, researcher bias
]

= Summary
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: This chapter includes the status of your thesis, a conclusion and an outlook about future work.
]

== Status
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Describe honestly the achieved goals (e.g. the well implemented and tested use cases) and the open goals here. if you only have achieved goals, you did something wrong in your analysis.
]

=== Realized Goals
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Summarize the achieved goals by repeating the realized requirements or use cases stating how you realized them.
]

=== Open Goals
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Summarize the open goals by repeating the open requirements or use cases and explaining why you were not able to achieve them. Important: It might be suspicious, if you do not have open goals. This usually indicates that you did not thoroughly analyze your problems.
]

== Conclusion
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Recap shortly which problem you solved in your thesis and discuss your *contributions* here.
]

== Future Work
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Tell us the next steps (that you would do if you have more time). Be creative, visionary and open-minded here.
]