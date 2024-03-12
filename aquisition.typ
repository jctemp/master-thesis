#set cite(style: "alphanumeric")
#set heading(numbering: "1.1")
#show heading: set block(below: 1.2em, above: 1.75em)

= Symmetric Normalization (classical registration approach)

- current used approach for regsitration

Find a spatiotemporal mapping, $phi.alt in italic("Diff")_0$ such that the cross-correlation between the image pair is maximized.

- intergration of $(d phi.alt(bold(x),t)) / (d t)$ yields $phi.alt(bold(x),t)$ @SyN2008 @SyN2010 @Dt2022
  - vector field is $phi.alt(bold(x),1) - x$ with $t in [0,1]$ describing the complete time span @SyN2010
- principle: not relying on fixed and moving image
- optimise a functions $phi.alt$ using ordinary differential equation (o.d.e) that describe the transformation path from A to B; derive $phi.alt^(-1)$
@SyN2008 @SyN2010 

- used for lung registration
- implementation using done using Advanced Normalisation Toolkit (ANT)
- compute time is outragous (hours)
- consideration of spatiotemporal features benefits registration
- fully automatic
- provides satisfactory registrations with acceptable errors
@SyN2010

- optimisation goals:
  - increase similarity
  - enforce symmetry

= Diffeomorphism

- one can describe the images with points $(x,y,t)$
- we can construct an Euclidean metric space $M subset RR^n : (M, d)$ where $d$ is a distance function
- accordingly, we have a set of points $M = {(x,y,t)}$
- a chart $(phi, U)$ with an open set $U$ on $M$ and $phi : U -> V : V subset RR^m : m = n + 1$
- $phi$ is a homeomorphisms (preserves topology; $C^0$) and the more constraint form a diffeomorphism (differentiable and smooth; $C^k : k > 0$)
- now, we use the set of charts representing an image along width and height and covering the manifold 
  $M = union.big_(t in I) U_t$ defining the atlas $cal(A)$ on $M$, essential is that 
  $(phi_i, U_i), (phi_j, U_j) : U_i sect U_j eq.not emptyset : i eq.not j$
- we need to find a transition function $psi: phi_b compose phi_a^(-1)$ which maps an image a to an image b
- because $phi$ is homemorphic and diffeomorphic, we get a $psi$ which is also homemorphic and diffeomorphic
- essential is that the atlas $cal(A) = {(phi_i, h_i)_(i in I)}$ is a $C^k$-atlas, meaning any two charts are 
  $C^k$-smoothly compatible. Further $cal(A)$ is maximal, meaning for any other $C^k$-atlas $cal(B)$, we have 
  $cal(A) subset.eq.not cal(B)$
  
- *note:* chaining multiple of transition functions allows us to transform an image A to an image B

The machine learning model approximates the transition funtion with these properties.
Later understood as cycle concitency

---

- for image registration these are essential properties as this avoids singularities @SyN2010
- further the order of fixed an moving image is irrelevant @SyN2008
- diffeomorphism belongs to group of homeomorphisms that preserve topology
- minimize temporal variability 
- gives desirable properties regarding time series aligment
@Dt2022

= Current landscape 

- MR imaging is superior for examining anatomical features
- significance in many discplines in medicine 
- registration uses dissimilarity between images
- traditional registration techiques, e.g. A.N.T. are optimisations problems with suboptimal computational requirements
  and limitations regarding complex deformations
- all open datasets for lungs are CT-scans => no comparability with public data
@reviewA

- registration uses primarily CNNs
- show significant higher registration success compared to traditional methods,
  even with only intensity-based unsupervised methods
- diffeomorphic deformation is already a metric
- employ U-Net @unet2015 architectures
- used CNN and RNN to train for image and text analysis, actor-model???

@reviewB

== Convolutional Networks

- similarity metrics
  - intensity-based: mean square distance, sum-of-square distance, (normalised) mutual information, (normalised) 
    cross-correlation
  - modern alternatives: structural similarity index measure, peak signal-to-noise ratio, target registration error
  - work well for mono-modality (MRI to MRI)

- similarity-based unsupervised deep learning: VoxelMorph and CycleMorph
@reviewA

- DIRNet an unsupervised deformable image registration network
- idea: directly learn image registration based on similarity measure
- end-to-end trained and can register images non-iteratively
- convolutional network to analyse local information based on its representations of image features
- spatial transformer generates displacement vector field
- CNN uses avg pooling, elu, batch norm and linear output (regressor)

@dirnet2017

- prior work is DIRNet
- assumptions: images are affinely aligned, single channel
- VoxelMorph is a architecture agnostic approach
- uses a CNN (U-Net @unet2015) to learn image features
- spatial transformer network with resampler to generate warped image, used for backpropagating the error @stn2016 
- loss function is mean square error or cross-correlation
- add vectors of displacement field to encourage smooth displacement
- compared to ANTs SyN and NiftyReg
  - showed similar registration accuracy
  - needs comparison to GPU implementation (close the gap between DL method and classic)
  - still DL method requires one minute => 150 time faster then SyN
  - DL method can work effectivl with small sample sizes (100 training images)

@voxelmorph2019

- take the idea of classic methods that generate diffeomorphic deformable mapping (e.g. SyN)
- unsupervised method trained by minimising loss between deformed and target image
- not imposing cyclic consistency leads to degenerated mapping (e.g. folding)
- Two networks $F: X -> Y, R: Y -> X$ where $F tilde.eq R^(-1)$
- spatial transformer layer used for warping and resampling @stn2016
  - network is differentiable
  - can be used to backpropagating error
- loss function
  - registration: local cross-correlation, l2-loss
  - cycle: $X -> hat(Y) -> tilde(X)$ and it must be $X tilde.eq tilde(X)$
  - identity: stationary regions should not be changed, identical image = no deformation
- method can be extened to multiscale registration with global and local deformation 
- compared to ANTs SyN, VoxelMorph and MS-DIRNet
  - better DICE and fewer non-positive values in Jacobian matrix compared to other DL methods 
  - multiscale approach robustly deals with large volume registration problems

@cyclemorph2020

- regrading CNN's VoxelMorph is state-of-the-art
- problem convolutions only capture local information, therefore fail to model long-range dependencies
- idea: introduce attention mechanism to have a AttentionVoxelMorph network
- attends spatial and coordinates of voxel to capture relationships 
- achieve smoothness through spatial gradient and bending penalty term
- performed better, however, not clear if it is purely due to attention (unclear if loss function s different)

@davoxelmorph2022

== Generative Adversarial Networks

- GAN-based unsupervised framework (cGAN, cycGAN) 
- promising results in cross-modality
- questioned is the validity of output accuracy
@reviewA

- for the loss function
  - normalised mutual information: matches joint intensity distribution of two images 
  - structural similarity index metric: image similarity based on edge distribution and other landmarks
  - VGG16 L2 distance: pretraind VGG16 feature maps - semantic information compared on different levels
- cycGAN achieve cycle consistency using two functions $G: X -> Y$ and $F: Y -> X$ with the domains $X, Y$; $X$ are the
  moving images and $Y$ are the fixed images $L_"cyc" (G,F) = E_x ||F(G(x)) - x|| + E_y ||G(F(y)) - y||$
- $L_"cyc"$ is weighted with a $lambda$
- the method does not use regularisation 
- paper indicated exceptional generalisation, flexible regarding mono- and multimodality
  - lacks comparison to other networks; CNNs and transformers

@gan2019

== Hybrid-Transformer

- experimental results showed better performance compared to VoxelMorph
- concept is to learn spatial transformations between images
- TransMorph has diffeomorphic spin-offs
- ViT-V-Net, TransMorph 

@reviewA

- aims to solve ConvNet's limitations regarding long-range spatial relations
- in DL there is a trend regarding self-attention
- use volumes to exploit spatial correspondence (expensive)
- hybrid-architecture
  - ConvNet's encode images to high-level features
  - high-level features get positional embeddings
  - passed through the ViT, V-Net decoder @vnet2016 applied with skip connections
- Results
  - dice comparison with Affine alignment, SyN, VoxelMorph
  - showed an increase on various anatomical structures

@vitvnet2021

- stressed the importance of diffeomorphic image registration
- before registration, moving image is affinely-aligned
- use vision transformer for encoder and CNN as decoder 
- used length of spatial gradients and bending energy to penalise non-smooth deformations
- on the heavy-side regarding parameters (approx. 45M)
- results 
  - TransMorph has moderate training, computationally not to hungry
  - CycleMorph is in computational worse, very training intensive due to cycle consistency
  - VoxelMorph is in relative small margin better 
  - ConvNet's produces smaller displacements than Transformer-based approaches
  - Swin Transformer architecture outperforms other transformer
  - Transformer achieved better scores than ConvNet's

@transmorph2022

== Derived recommendation

- high computational efficiency and comparable accuracy compared with traditional methods
- Diffeomorphic Registration is essential for registrations
- all architectures are heavy (very large parameter size)
- large deformation of soft tissues (e.g. lung respiratory motion) challenging
- use L2 norm for normalisation

@reviewA

- end-to-end training of models
- incooperate expert knowledge
- one should not underestimate the influence of pre-processing and the effect on
  generalisation; drastic for CNNs
- bad performance indicator for wrong network architecture or bad input

@reviewB

== Comparison

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
  )
)

- CNNs are established @reviewA @reviewB
- Transformers are novel @reviewA but potentially better then CNNs
- GANs are promising with results but not well established + concerns regardin output validity @reviewA

= Setting

- high image hetrogenity e.g. scanners and patient with various disease introducing more variance
- MR images have non-stationary noise 
@SyN2008

- temporal and spatial variations during respiratory cycle
- currently using traditional methods implemented with advanced normalisation toolkit
- based-on registered pulmonary images perform diagnostic analysis (perfulsion and ventilation)
@Voskrebenzev2017

- solving equations for registrations is computationally expensive
- *goal:* replace solving ODE with neural network for faster registration
- *requirement:* cannot sacrifice accuracy of current method
- *optional:* another approach than STN

= Method

#bibliography("references.bib")
