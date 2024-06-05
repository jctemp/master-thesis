= Background

This chapter aims to provide foundational knowledge about machine learning and image registration.
We start with machine learning, exploring the subarea of deep learning. 
We then discuss medical image registration and its connection to deep learning in the current landscape.

== Machine learning

Machine learning is an essential tool in modern computer science, offering numerous methods for analysing and processing large amounts of data. 
The critical difference to classical software engineering is the highly declarative nature of these methods.
Declarative means that engineers are not required to define explicit instructions to solve a given problem; the method can discover a sufficient solution, whether exact or approximate.
Accordingly, we define machine learning as follows:

Machine learning refers to techniques for uncovering valuable insights from data.
These techniques involve sophisticated mathematical models that map input to output sets.
A model $f: A -> B$ drawn from a hypothesis space $cal(H)$ has an internal state that captures complex relationships within the domains.
By exposing the model to sample data and evaluating its solution against an objective function, the model continually optimises its internal state towards the goal defined by the objective function.
As a result, it adapts behaviour, which can be thought of as learning @mitchellDisciplineMachineLearning2006 @dehouwerWhatLearningNature2013 @lecunDeepLearning2015.
To find a suitable function, we require a learning algorithm $cal(A): D -> f$ that provides a mapping from a dataset $D subset.eq A$ to the desired function $f in cal(H)$. Multiple strategies exist to derive such a function, such as supervised and unsupervised learning @goodfellowDeepLearning2016.

=== Neural networks

In the context of this thesis, we will work with neural networks inspired by neuroscience @goodfellowDeepLearning2016.
Nowadays, one recognises these networks as universal approximators for any measurable function @hornikMultilayerFeedforwardNetworks1989, achieving statistical generalisation @goodfellowDeepLearning2016. They are composed of computational units called neurons.
These neurons take some input vector $bold(x)$ and linearly transform them using weights $bold(w)$ and a bias $b$.
Note that we utilise $bold(theta)$ to represent the parameters of a neural network.
The linearly transformed output then passes through a non-linear function, e.g. a sigmoid, to project the new high-dimensional input space to make it potentially linearly separable @lecunDeepLearning2015.
Accordingly, finding the optimal projection requires a systematic strategy given by a learning algorithm $cal(A)$ like gradient descent. 

=== Gradient descent

Gradient descent is a numerical optimisation technique. 
Given a neural network representing a function $f(bold(x); bold(theta))$, we aim to minimise the numerical output of a differentiable objective function $cal(F)$ @goodfellowDeepLearning2016.
The objective function measures the output $bold(hat(y))$ based on domain-specific constraints.
Gradient descent computes the partial derivatives $nabla bold(theta)$ to obtain the change concerning the global function's output @lecunDeepLearning2015. 
Formally, the optimisation process can be expressed as:

$
f^* = op("argmin", limits: #true)_f EE_{bold(x) ~ p_"data"} cal(F)(bold(x))
$<eq-optim>

The equation @eq-optim states that we desire to uncover a $f$ in the hypothesis space that minimises $cal(F)(bold(x))$ over all data points given $p_"data"$, a probability distribution.
Note that the hypothesis space is a function space containing all sound functions with the expected mapping. 

The $cal(F)$'s exact structure depends on the domain and task we aim to solve. 
Further, the data can provide ground truth values that can be used during optimisation.
Hence, we differentiate between two approaches for optimisation: supervised and unsupervised learning.

=== Supervised and unsupervised learning

The difference between supervised and unsupervised learning is the incorporation of ground truth data, which affects the structure of the dataset $D$ and the objective function $cal(F)$.
We assume a function $f: A -> B$ and aim to find an optimal mapping between the two sets.

For supervised learning, the dataset $D:= {(bold(x), bold(y)): bold(x) in A, bold(y) in B}$ consists of data points represented by tuples, providing information about the input and a desired output.
Accordingly, the learning algorithm $cal(A)$ will present a function $f$ providing an output $bold(hat(y)) approx bold(y)$ given $bold(x)$ as $P{bold(hat(y)) | bold(x)}$ @goodfellowDeepLearning2016.
The supervision signal is incorporated in the objective function $cal(F)(bold(x), bold(y))$, which is often acquired through manual labour.
Therefore, it is not applicable for all tasks. The objective function measures the error between output and ground truth, and we average the error score over all training examples @lecunDeepLearning2015.

In contrast to supervised learning, unsupervised learning does not require ground truth data @lecunDeepLearning2015. 
It aims to learn the probability distribution $p_"data"$ of a dataset $D:= {bold(x): bold(x) in A} space "and" space D subset.eq A$. 
Hence, it involves extracting distribution properties based on the data's structure @goodfellowDeepLearning2016 without a supervision signal.
Instead, unsupervised learning focuses on finding the data's best representation regarding penalties or constraints regarding $cal(F)(bold(x))$.

== Deep learning

=== Convolutional neural networks

=== Recurrent neural networks

=== Transformer

=== Vision transformer


