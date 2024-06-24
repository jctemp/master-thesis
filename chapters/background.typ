// #bibliography("../references.bib")
// #set math.equation(numbering: "1.1")

= Background

This chapter aims to provide foundational knowledge about machine learning and image registration.
We start with machine learning, exploring the subarea of deep learning. 
We then discuss medical image registration, aligning two or more images of the same scene or object, and its connection to deep learning in the current landscape.

== Machine learning

Machine learning is a subfield of computer science that provides many methods to analyse large data quantities.
The goal of these methods is to automatically find a solution, whether exact or approximate, for a given problem or task.
The solution is derived based on given data without explicit instructions, making it highly declarative.
From a mathematical perspective, one aims to find a mapping $f: A -> B$, describing the task where the domain $A$ and the range $B$ can be arbitrary @mitchellDisciplineMachineLearning2006 @lecunDeepLearning2015. 
An objective function $cal(F): (f, D) -> RR$ is then used to assess the quality of the function mapping, and based on this information, $f$'s internal state is iteratively modified to optimise its performance towards the objective function.
An iterative optimisation procedure is then applied to a learning algorithm $cal(A): D -> f$,  drawing a function candidate $f$ from its hypothesis space $cal(H)$, yielding the desired mapping based on sampled data $D$. 
Note that the hypothesis space is a set of all possible functions according to a learning algorithm $cal(A)$.
Furthermore, example $D$ is drawn from data distributions such as $p(A)$ or $p(A, B)$, determining the learning strategy (e.g., supervised or unsupervised learning), which we explain later.
As a final remark, the process of adapting the function's behaviour based on examples is what we refer to as learning @dehouwerWhatLearningNature2013.

=== Neural networks

Neural networks are one of the many learning algorithms used in machine learning.
They are inspired by neurons observed in neurological systems in nature @goodfellowDeepLearning2016
Neurons experience excitement (input), causing a reaction (output), which can be mathematically defined as seen in @neuron.
Nowadays, these networks are recognised as a universal approximator for any measurable function @hornikMultilayerFeedforwardNetworks1989, achieving statistical generalisation. 
A neuron has an input vector $bold(x)$, which undergoes a linear transformation with learnable weights $bold(w)$ and a bias $b$. 
The transformation projects the input to a new $n$-dimensional space passed to an activation function $sigma$.
The activation function is non-linear, like sigmoid, tanh and rectified linear unit (ReLU), which is crucial as it makes the output potentially linearly separable for other neurons @lecunDeepLearning2015.

$
a = sigma(bold(x) bold(w)^top + b)
$ <neuron>

Solely using a single neuron is insufficient for more complex problems like XOR @goodfellowDeepLearning2016.
Therefore, one employs layers defined in @layer to obtain a multi-dimensional intermediate representation $bold(a)$.
Layers consist of multiple neurons, contributing a feature to the new representation for the next layer.

$
bold(a) = sigma(bold(x) bold(W)^[l] + bold(b)^[l])\
$ <layer>

We describe the weights and bias with vectors and matrices, as in @wandb.
The weights matrix holds column-wise $n$ neurons and row-wise the weights for $n-1$ input features.
A vector is used to provide bias for each neuron.

$
bold(W)^[l] = mat(
 w_00, ..., w_(0n^[l]);
  dots.v, dots.down, dots.v;
 w_(n^[l-1]0), ..., w_(n^[l-1]n^[l]);
) in RR^(n^[l-1] times n^[l]), quad
bold(b)^[l] = mat(b_1, ..., b_(n^[l])) in RR^(1 times n^[l])
$ <wandb>

Note that the hypothesis space expands with the addition of neurons and layers, as the increased number of parameters and non-linear activation functions enable a wider range of discoverable functions.
We refer to the XOR problem @goodfellowDeepLearning2016.

=== Gradient-based optimisation

The neural networks have an internal state described by their weights and biases.
Additionally, we denoted all learnable parameters of a network with $bold(theta)$.
At first, the parameters are random, which we want to modify to meaningful values.
Thus, researchers employed the gradient-based numerical optimisation technique.

Given a network with parameters $f(bold(x); bold(theta))$, we aim to minimise the numerical output of a differentiable objective function $cal(F)$ @goodfellowDeepLearning2016.
Recollect that the objective function is the task the network aims to solve.
One can compute the partial derivatives $nabla bold(theta)$ for the parameters $bold(theta)$, with respect to the networks output $bold(hat(y))$ and the objective function $cal(F)$ @lecunDeepLearning2015.
These gradients are then used to adjust the weights to optimise the objective function's output.

For example, we can express the optimisation problem, as in @optimisation
The optimal function $f^*$ is found by changing the parameters $bold(theta)$.
The equation states that we search for parameters $bold(theta)$ that minimise the objective function over all data points given a probability distribution.

$
f^* = op("argmin", limits: #true)_bold(theta) EE_{bold(x) ~ p_"data"} cal(F)(f_bold(theta), bold(x))
$<optimisation>

The quality of the solutions found highly depends on how well the objective function models the real-world task.
A poor objective function leads to poor results.
Likewise, gradient changes that are too influential can cause the same problem.

=== Supervised and unsupervised learning

Supervised and unsupervised learning are two learning strategies relevant to this thesis.
We select one depending on the data's distribution and available information.
Accordingly, the core difference is the presence of ground truth data, affecting the structure of a dataset $D$ and the objective function $cal(F)$
Recall that the function $f$ has a domain $A$ and a range $B$ from which one can pull samples.

Supervised learning has ground truth data $D:= {(bold(x), bold(y)): bold(x) in A, bold(y) in B}$, which we indicate with tuples in the dataset.
Owing to this, an objective function can measure the difference between the function's output $bold(hat(y))$ and the target $bold(y)$ given the corresponding $bold(x)$ @goodfellowDeepLearning2016 @lecunDeepLearning2015.
The supervision signal enables training based on expected outputs often acquired through manual labour.
Therefore, it is not applicable for all tasks.

In contrast to supervised learning, unsupervised learning does not require ground truth data @lecunDeepLearning2015. 
It aims to learn the probability distribution $p_"data"$ of a dataset $D:= {bold(x): bold(x) in A}, space D subset.eq A$. 
Hence, it involves extracting distribution properties based on the data's structure @goodfellowDeepLearning2016 without a supervision signal.
Instead, unsupervised learning focuses on finding the data's best representation regarding penalties or constraints regarding $cal(F)(bold(x))$.

== Deep learning

Deep learning is a subfield of machine learning that builds upon the concept of neural networks. @lecunDeepLearning2015 @goodfellowDeepLearning2016.
One characterises deep learning models through multiple stacked layers forming a deep network.
Note that each layer can consist of many neurons, like the multilayer perceptron.
This layered architecture allows deep neural networks to learn complex functions by gradually extracting higher-level features from the input data @lecunDeepLearning2015.
The training process employs gradient descent, usually minimising an error modelled by an objective function @lecunDeepLearning2015 @goodfellowDeepLearning2016.
Increasing the number of layers has three effects: capacity, distributed and composed representations.
First, deep neural networks can have many mutable parameters, which provide enough capacity to approximate high-dimensional and non-trivial measurable functions @hornikMultilayerFeedforwardNetworks1989.
Further, multiple neurons work together to represent a single feature, and natural signals often exhibit a hierarchical structure that can be captured using composed representations @lecunDeepLearning2015.

Next, we briefly overview the most relevant deep learning concepts and architectures.
We will start with convolutional neural networks that process grid-structured data and then move on to recurrent neural networks for sequential time-based signals.
Finally, we explain transformers and vision transformers.

=== Convolutional neural networks

Convolutional neural networks (CNNs) are specialised in processing grid-like structures, such as images, by extending the concept of neurons (learnable weights and biases) to filters.
A filter is simply a matrix with learnable values.
One uses the filter to compute a discrete convolution regarding an input.
Convolving with the filter over an image allows the CNN to capture location-invariant local statistics represented through the filter weights (matrix) @lecunDeepLearning2015 @goodfellowDeepLearning2016.
Besides location-invariance, filters create a sparse connection between layers, so weights in the current layer do not depend on all activations of the previous layer.
Another benefit of using filters in neural networks is parameter sharing. 
This means the same set of weights is used across all input pixels, allowing each pixel to affect the weights. 
This parameter sharing decouples the model parameters from the input size, allowing for more efficient learning.
Filters allow the model to learn meaningful features like lines and curves.
Combining multiple filters per layer enables CNNs to distribute representations across units and enhance generalisation @lecunDeepLearning2015.
Finally, CNNs use so-called pooling layers, which provide a statistical summary of a local neighbourhood, improving translational invariance @goodfellowDeepLearning2016.

The theoretical advantages of CNNs were proven multiple times  @NIPS2012_c399862d, @heDeepResidualLearning2015, @ronnebergerUNetConvolutionalNetworks2015, @milletariVNetFullyConvolutional2016.
They outperformed traditional computer vision methods by a wide margin. 
As their width and especially depth increased, CNNs became more capable of successfully extracting semantic information for classification @NIPS2012_c399862d and later other tasks like segmentation and registration @ronnebergerUNetConvolutionalNetworks2015, @chenTransMorphTransformerUnsupervised2021.
However, deep networks introduced the issue of vanishing and exploding gradients caused by the numerical instability of partial derivatives.
Consequently, researchers @heDeepResidualLearning2015 presented the concept of residual connections, which leads to more stable gradients and combats another issue called the degradation problem (the insufficiency of a model to learn identity function).
Meanwhile, pixel-level detail tasks like segmentation @ronnebergerUNetConvolutionalNetworks2015 were in demand, so the authors presented a new architecture called UNet.
They split the architecture into three parts: contracting, expanding, and skipping paths.
With each step, the contracting path applies convolutions and pooling to create a more high-level feature map.
The expanding path uses transposed convolutions paired with unpooling layers to restore the original input size.
Further, the expanded output and the contracted input for each corresponding step are merged via skip connections, fused to a more meaningful representation combining lower-level and higher-level information.
Accordingly, a network can make accurate pixel-wise predictions.
Extending the concept of UNet, other researchers @milletariVNetFullyConvolutional2016, @chenTransMorphTransformerUnsupervised2021 presented the VNet, which allows segmentation on volumes, like voxel data or videos.

=== Recurrent neural networks
@gilesDynamicRecurrentNeural1994
Recurrent neural networks (RNNs) are a paradigm for processing sequential data of arbitrary length @goodfellowDeepLearning2016, @gilesDynamicRecurrentNeural1994.
They have a hidden state $h^((t))$ to store information over time, where the network can control the influence of the input signal on the memory through concepts @gilesDynamicRecurrentNeural1994, @hochreiterLongShortTermMemory1997.
Furthermore, the hidden state summarises past and current information in a lossy fashion.
A recurrent unit uses the current input signal and hidden state to derive a new hidden state $h^((t+1)) = f(W h^((t)) + U x^((t)) + b)$, maintaining the most relevant information for the next recurrent unit @goodfellowDeepLearning2016.
Note that $W$ is the recurrent weight matrix, $U$ is the input weight matrix $x^((t))$, and $b$ is the bias. 

Interestingly, an RNN is equivalent to a layered network because one can unfold the recursive definition $h^((t))=f(h^((t-1)), x^((t)); theta)$ constructing a directed acyclic graph @gilesDynamicRecurrentNeural1994, @goodfellowDeepLearning2016.
Hence, a recurrent unit is similar to a layer keeping another type of state.
Moreover, we can apply the existing training method gradient descent, owing to a graph representation @gilesDynamicRecurrentNeural1994.

The ability to process arbitrary long sequences comes with two drawbacks.
First is the vanishing or exploding gradients @goodfellowDeepLearning2016.
We have this problem because one reapplies the same function with the same weights over multiple time steps, leading to unpredictable non-linearity.
Furthermore, one can simplify the problem by ignoring the input signal and non-linear functions, giving the following formulation $h^((t)) = W h^((t-1))$.
Assuming the weight matrix of a recurrent unit can undergo eigendecomposition, we get $h^((t)) = Q Lambda Q^(-1) h^((t-1))$ where $Q$ are the eigenvectors and $Lambda$ the eigenvalues.
Repeatedly applying the function with Eigenvalues other than one will exponentially increase or decay the initial state, illustrated by rearranging the previous definition $h^((t)) = Q Lambda^t Q^(-1) h^((0))$.
Hence, RNNs struggle to learn long-term (long-range) dependencies efficiently.
Of course, concepts like long-short-term memory @hochreiterLongShortTermMemory1997 exist to address the issue.
Nonetheless, areas like natural language processing adopted a new paradigm called attention-based learning due to its better parallelisability while allowing the modelling of long-term dependencies.

=== Transformer architecture

The transformer architecture, popularised in 2017 with the paper "Attention is All You Need" @vaswaniAttentionAllYou2017, addresses the limitations of RNNs in modelling long-range dependencies and has surpassed them in natural language processing tasks.
The crucial innovation in the transformer was the multi-head self-attention.
Other concepts contributed to its success, such as residual connections @heDeepResidualLearning2015, layer normalisation, positional encoding, and learnable embeddings.

In natural language processing, we want to understand the syntactic and semantic relationship between various words in a sentence.
First, one splits the sentence into so-called tokens (numerical representation) describing different parts of the sentence.
Next, these tokens are transformed into an embedding space, assigning a semantic meaning.
Before processing the sequence, each embedding undergoes a positional encoding to indicate the token's position in a sentence.
Positional encoding is required because self-attention is position-invariant, and a word's meaning can change depending on its context.
Finally, one can pass these embeddings with positional encodings to a transformer block, computing the self-attention and applying a multilayer perceptron to add non-linearity to get the new embedding.
The self-attention module is essential to encode relationships between the embeddings during this process.

Equation @self-attention shows the self-attention.
We require three matrices: query ($Q in (n, d_k)$), key ($K in (n, d_k)$), and value ($V in (n, d_v)$), where $d_k$ and $d_v$ are the dimensions of the embeddings.
The matrix multiplication $Q K^tack.b$ gives the row-wise cosine similarity, examining the alignment of a query to all the keys.
In other words, it checks the semantic relationship of n words in a sentence.
The denominator $sqrt(d_k)$ is a normalisation factor to prevent small gradients in the softmax function.
The row-wise softmax function normalises the score for each query across all keys, indicating relevance in the form of attention weights.
Finally, we can compute the contribution of each value embedding.

$
"SelfAttention"(Q,K,V) = "softmax"((Q K^tack.b)/sqrt(d_k)) V
$<self-attention>

The equation @multi-head-self-attention shows multi-head self-attention, which leverages local representations to improve attention capabilities and the possibility of learning relevant relations.
It utilises learnable projection matrices $W^Q_i, W^K_i in (d_"model", d_k),$ and $W^V_i in (d_"model", d_v)$ to get a local representation of queries ($Q in (n, d_"model")$), keys ($K in (n, d_"model")$) and values ($V in (n, d_"model")$).
Note that $d_"model"$ is the dimensionality of the input embeddings to the transformer, and $d_k$ and $d_v$ are dimensions for the embeddings within the multi-head self-attention module.
Then, one computes the self-attention over these representations.
Ultimately, the module concatenates the results, followed by a linear projection ($W^O in (h d_v, d_"model")$) to get the final output.

$
"MultiHeadAttention"(Q,K,V) = op("concat", limits: #true)_(i in I_h) ["Attention"_i (Q W^Q_i,K W^K_i,V W^V_i)]W^O
$<multi-head-self-attention>

The apparent advantage is finding relations between distant words through attention, thus modelling long-range dependencies.
Computing the attention for a sequence are trivial matrix multiplications, which are highly optimised on modern graphics processing units.
On the other hand, transformers lose the flexibility of variable input length because they have no recurrence, which is combated with larger input sequences. 
However, this leads to a computational burden due to each token attending to all tokens. 
Moreover, transformers do not have an inductive bias like CNNs and RNNs, which capture local or sequential patterns @cordonnierRelationshipSelfAttentionConvolutional2019 @battagliaRelationalInductiveBiases2018.
Depending on the context, one can exploit the bias to improve the model's performance regarding data requirements or training time.
Accordingly, attention-based learners need special care to learn these structures, which may involve more training examples or other architectural and training approaches.

Nevertheless, the attention mechanism is a potent concept to describe complex data attributes and their relationships.
In addition to working with text, researchers introduced vision transformers, which can process image data, opening the possibility for multi-modal processing.

=== Vision transformer

The vision transformer @dosovitskiyImageWorth16x162020 is the adaptation of a transformer for computer vision tasks.
Recollect that one cannot process a two-dimensional image directly, as the transformer solely consumes a linear arrangement of tokens.
Consequently, the authors proposed a strategy to convert an image into tokens.
An image is divided into equally sized patches, then linearised and projected into a learnable $d$-dimensional embedding space.
One adds a two-dimensional positional encoding to preserve the spatial information as the information is lost during the linearisation of an image.
Dosovitskiy et al. demonstrate on-par or exceeding performance in classification compared to state-of-the-art CNNs.
Note that transformer self-attention heads can imitate convolutions, leading to CNN-like behaviour @cordonnierRelationshipSelfAttentionConvolutional2019 and explaining the on-par performance.

Nonetheless, the vision transformer @dosovitskiyImageWorth16x162020 is not a functional general backbone network like a CNN, owing to the patch generation needing to generate smaller patches to capture local information, which is necessary for pixel-level tasks like segmentation.
Reducing the patch size would lead to a quadratic increase in tokens and computing self-attention, which is quadratic in runtime and unattainable with current computing.
An example to make the problem more explicit: we assume a patch size of four and a low-resolution image like $256 times 256$.
This would already result in $16384$ tokens, which requires a considerable context length by current standards.
Conversely, self-attention requires modification to work with large-scale input.

Liu et al. @liuSwinTransformerHierarchical2021 proposed the shifting-window hierarchical transformer (swin transformer) to combat the issue of quadratic token increase. 
The key concept is the switch to a local attention model, in which the image tokes are partitioned into disjoint windows, and self-attention is estimated over these windows.
It decreases the computational complexity of attention for an image, diminishing it to linear time $O(M * N)$ where $M$ is the size of the local window, assuming that $M$ is significantly smaller than $N$; otherwise, we would have normal multi-head self-attention.

The authors subdivided the swin transformer into three steps: patch generation, first and consecutive stages.
Foremost, analogue to ViT, images with the dimensions $(H, W, C)$ are divided into patches of size $P times P$.
By flattening each patch, it is converted to a linear vector, resulting in dimensions $H/P times W/P times (P dot C)$.
Compared to ViT, a patch $(i,j)$ now belongs to a local window of size $M$ (e.g. $4 times 4$), respecting the spatial structure crucial for self-attention.
Next, the first stage projects the linearised patches into a $d$-dimensional embedding space, giving a vector with the dimensions $H/P times W/P times D$.
Subsequently, each local window is passed to two consecutive swin transformer blocks, analogous to a transformer block.
The sole distinction is that the first block has a window-based multi-head self-attention, and the second block uses a shifted window-based multi-head self-attention, which we explain in the following paragraph.
The output of the previous stages undergoes patch merging.
Patch merging takes $2 times 2$ window of tokens and concatenates them into a vector of length $4D$.
This vector is passed through a linear transformation, halving the dimensions to $H/(2 P) times W/(2 P) times 2D$.
After that, the swin transformer blocks are applied again.

Recollect that traditional multi-head self-attention @vaswaniAttentionAllYou2017 computes self-attention across all tokens.
Thus, we can recognise it as a global self-attention computation.
To address the issue of global self-attention, Liu et al. @liuSwinTransformerHierarchical2021 introduced the concept of window-based multi-head self-attention (see figure @window-based-self-attention).
Recall that a window comprises $M times M$ tokens representing a neighbourhood shown by the red rectangles in the figure.
The authors suggest evaluating multi-head self-attention in these neighbourhoods.
Additionally, they employ a shifted window multi-head self-attention to relate patches across window boundaries, as demonstrated on the figure's right.
Otherwise, we would ignore integral semantic relationships between windows.
Combining the local self-attention with hierarchical processing, one effectively estimates global self-attention.
Finally, the shift causes exceptions at the image boundaries because windows are partially filled with tokens.
Handling partial windows increases the number of attended windows, and the question arises of how missing tokens are semantically represented.
Therefore, Liu et al. introduced a cyclic and a reversed cyclic shift to simplify computation and avoid exception handling.
One shifts the image $M/2$ in both axes and passes it to the window-based multi-head self-attention module, providing cross-boundary attention.
Accordingly, boundary patches assemble their windows.
The shifted-window self-attention balances local attention and information flow across windows.

#figure(
  image("../figures/w-msa.png"), 
  caption: [
 The figure depicts the window-based multi-head self-attention (W-MSA) and shifted W-MSA.
 ] 
)<window-based-self-attention>

All these changes allow the swin transformer to process larger inputs and obtain local semantic information, resulting in an improved contextual representation.
Furthermore, the output format is equivalent to CNNs, enabling the replacement of CNNs in existing computer vision architecture.
Note that semantic information is encoded differently but is irrelevant for decoder networks.

== Image registration

=== Medical application

=== Traditional techniques

=== Deep learning techniques

