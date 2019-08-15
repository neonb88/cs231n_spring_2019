# CS231n_Spring_2019

Solutions to [Stanford CS231n Spring 2019 Course](http://cs231n.stanford.edu/2019/) assignments.

If there are any mistakes, or you have questions about solutions, please contact me via github.  My e-mail is [firstname][lastname]@gmail.com (my name is Nathan Bendich).  I apologize if I do not immediately get to you.  I would love to give an in-depth walkthrough of every detail to every person who contacts me, but unfortunately time probably will not permit that.


## A few resources I found helpful:
  1. Visualizations of Convolution's backward pass:  https://github.com/vdumoulin/conv_arithmetic
  2. Backpropagation notes from Karpathy: http://cs231n.github.io/optimization-2/
  3. GAN Training:  https://github.com/soumith/ganhacks
  4. Deep Learning Reading List: http://deeplearning.net/reading-list/  
  5. Taking partial derivatives (called "Gradients" but more technically should be called "Jacobians") by hand: http://cs231n.stanford.edu/vecDerivs.pdf
  6. Softmax & 2-layer NN derivation by Karpathy:  http://cs231n.github.io/neural-networks-case-study/

### Things I wish I'd known when starting the lectures:
As far as I can tell, Xavier initialization is not so necessary any more.  Batch/Group/Instance/Layer normalization fixes the old problems Hinton et al. had with properly initializing neural networks.  Dropout still seems very relevant.  Nowadays skip connections in ResNet, ResNeXT, etc. make the early lecture content about regularization less relevant/totally outdated



### Brief Summaries (I apologize if these are more geared towards my personal commercial and research interests):
##### Also, I wrote these right after taking cs231n; obviously a more experienced engineer/academic like Geoffrey Hinton or Andrej Karpathy will have a better perspective on what's important, etc.
  0.  Like a baby's brain, a neural network learns from "experience" (Training Data for a Neural network; Days of life and events for a baby human).  A person may see 10 dogs, misidentify 5 cats, zebras, horses, etc. as dogs, and develop an abstract idea of what it means for a thing to be a "dog."  A neural network learns these ideas in big groups of numbers ("weights," more technically), but I find the analogy of deep learning to human learning helpful, and you may too.


________________________________

  cs231n Lecture 2:
________________________________

    Basically, simpler machine learning methods don't work for image data.  Seeing and making sense of the world is actually pretty damn difficult, and basic hard-coded computer algorithms don't do it well.  The reasons you as a person "see" so well might be 1) evolution weeded out a lot of the apes that didn't see very well so all your direct ancestors had pretty decent vision,  2) you've spent a large portion of your life learning what things in your field of vision are called   and b) how they're related to the words that are in people's heads, come out of their mouths, and are written down on the internet and pieces of paper.
    Technical details are all in the lectures; I won't go into them here.  They are important, but in order to understand them, you really just need to sit down with the code and go through each and every last detail.  Reading a bunch of text won't really be enough.
  1.  Affine layers
    a.  Linearity
  2.

________________________________

  cs231n Lectures "meta notes"
________________________________

  Besides recording the timestamps everywhere, I also found it helpful to just go to the slides (ie. http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture08.pdf), and search for text.
    ie. query "Dropout"


________________________________

  cs231n Lecture 6 ("Training Neural Networks I"):
________________________________


    All the best takeaways are at the END of Serena's Lecture 6.   (she summarized EVERYTHING)

    Serena talks about activation functions until

   Overview:
    (this slide is shown at time 0:04:39)
    Part 1:  
      1.  Activation Functions
      2.  Data Preprocessing
      3.  Weight Initialization 
      4.  Batch Normalization
      5.  Babysitting the Learning Process
      6.  Hyperparameter Optimization

    (part 1 is Lecture 6;  
     part 2 is Lecture 7)


     1.  Activation Functions
      starts at time  4:49
     2.  Data Preprocessing
      starts at time 27:17

     3.  Weight Initialization 
      starts at time 34:22
        Xavier init   fixes a lot of the problems.

     4.  Batch Normalization
      starts at time 49:00
        If we want unit gaussian activations, we can just make them unit
        From cs231n Assignment 2, BatchNormalization.ipynb:  "Batch normalization has proved to be effective in making networks easier to train, but the dependency on batch size makes it less useful in complex networks which have a cap on the input batch size due to hardware limitations."
          So we use layer normalization.  Layernorm does the same thing, but for a single training example:
            """
             In this paper, we transpose batch normalization into layer normalization by
             computing the mean and variance used for normalization from all of the summed
             inputs to the neurons in a layer on a single training case. Like batch normalization,
             we also give each neuron its own adaptive bias and gain which are applied after
             the normalization but before the non-linearity. Unlike batch normalization, layer
             normalization performs exactly the same computation at training and test times
            """
            - Ba, Kiros, & Hinton, 2016


            """
            Layer Normalization:
              Batch normalization has proved to be effective in making networks easier to train, but the dependency on batch size makes it less useful in complex networks which have a cap on the input batch size due to hardware limitations.
              Several alternatives to batch normalization have been proposed to mitigate this problem; one such technique is Layer Normalization [2]. Instead of normalizing over the batch, we normalize over the features. In other words, when using Layer Normalization, each feature vector corresponding to a single datapoint is normalized based on the sum of all terms within that feature vector.
            """




     5.  Babysitting the Learning Process
     6.  Hyperparameter Optimization

  ________________________________

    TODO during training   (Serena lays out all the steps at 1:04:44.):

      Basic sanity checks:
      0.  Initial loss value (is loss(X,W,   epoch=0) correct?)
      1.  Checking initialization:
        a.  Turn off the regularization.
        b.  SMALL small sample of training dataset,    
          1)  get loss to 0.  (sanity check)
          2)  Training acc  should go to 1  on the small set


      2.  Learning Rate  
        a.  (now turn on regularization)
          Wait, is it principled to use regularization on NNs?  (JJohns alluded to this, saying that Dropout was a more principled version of regularization for NNs than L2/L1 reg)
            Read paper(s) (ie. the Dropout paper) to tell
              Actually, I might have to dive into the f***king code / weights and do it that way.   Hrmph.
        b.  Get the right learning rate
        c.  Too small => loss to 0
        d.  Too big => loss to +\infty   (NaNs)
        e.  (Broad)  Run for a few epochs  over many LRs   (broad)

      3.  Sample other hyperparams randomly rather than uniformly???  I'm still not sure what she means here.  Like "change the bin size regularly so you don't miss some magical hyperparameter setting????"
        a.  But what range of hyperparams should we sample from, Serena?
        b.  I think the random noise on top of np.logspace() is decent.
          1) max_lr8 = 3e+0 #1e+2
          min_lr8 = 3e-2 #1e-8
          n_lr8s  = 25 #17  #11 #100

          l_r8s=np.logspace(
              math.log10(min_lr8),
              math.log10(max_lr8),
              n_lr8s
          )
          # noise
          l_r8s+= (np.random.random(
              l_r8s.shape,
          ) * 3e-2)   # BAD: the noise on the lower end is more significant than on the upper end.

      4.  Graph it!
        a. acc
        b. loss




      Problems:
        a.  The loss starts high & flat, goes down sharply at a certain point
          1.  The issue is probably bad initialization =>    Fix it by... merely initializing again?  Xavier Init?
        b.  
        c.
        d.



    Initialization:
      Understand Xavier initialization
        Doesn't work with ReLU
          B/c ReLU kills half the neurons (whenever they're in the "zero regime").
      Xavier does (I think, roughly speaking)   "Calculate the mean & variance @ each layer & FORCE it to be a unit Gaussian"

      Same as batch norm?
        mostly in CNNs, not Recurrent

        Compute the empirical mean & variance   across each layer & batch
        Also works for Convolutional layers   AND Fully Connected layers
      Xavier will be done in HW 2

      Xavier changes INPUTS to each layer to be std Gaussian
        Before or AFTER the nonlinear activation func (ie. tanh, ReLU, PReLU, Leaky ReLU, etc.)?

    Backprop & 
    Activation functions are LESS important, but also still very important.
      I think leaky ReLU is the best, based on what Serena was saying.
      Read up on why saturation might actually be GOOD
    Backprop is pretty much the key to the whole learning process.

    Dimensionalities & what the linearity MEANS





  ________________________________

    Batch Normalization
      After FC / conv layer
      Before nonlinearity

      Can backprop through this batch normalization (just take a partial derivative and multiply by it)

      Batch norm lets the network LEARN the scaling and shifting?
        Gamma and Beta
          learned during training (not man-set hyper parameters)
      But is this the point of batch norm or, are we supposed to just use stddev and mean   to scale and shift?


      I think Serena mispeaks:
        if inputs to a layer aren't Gaussian, there's no way a simple shift and scale would make them Gaussian-ly distributed.  Imagine points -1, 0, 1, 2, and 3 (a reasonable approximation of a uniform distribution).  if we shift by the mean (-1 to all); [-2, -1, 0, 1, 2], and scale (divide) by the variance (10/5 = 2)  (10 b/c we square the differences from x_i to mean mu; 2**2 + 1**2 + 0**2 + ...), we get [-1, -0.5, 0, 0.5, and 1] (still uniformly distributed)
  ________________________________











________________________________


________________________________

  cs231n  Lecture 7 ("Training Neural Networks II"):
________________________________


    Assignment 2 is the longest one.


    Most of Assignment 2 is only CPU.
      Cheaper!  :)
    While working on Assignment 2, look back at the slides at time 8:07

    Coarse => fine search
      initial deltas should be BIG over a big space
    Then we narrow in
      And we learned this experientially/empirically anyway

    Usually changes in 1 hyperparameter don't affect the others' optimal values THAT MUCH.


    This lecture covers the fancier optimization algorithms
      1.  Momentum
        a.
      2.  RMSProp
        Slows down the learning rate later in the learning process
        Sample code:
          cache = decay_rate * cache + (1 - decay_rate) * dx**2
          x += - learning_rate * dx / (np.sqrt(cache) + eps)
      3.  AdaGrad
      4.  AdaMax
      5.  AdaM
      6.
      7.
      8.
      9.
      10.
      11.
      12.
      13.
      14.
      15.
      16.
      17.
      18.


    Other types of regularization (at time 14:15)
      Problems with SGD:
        1.  Loss may have "high condition number"
          Meaning == "Much more sensitive to one input variable    than the other"
            This problem is much much worse in very high dimensional space than in R^3.
          "Taco shell problem"
            AKA "angled half-pipe" (Tony Hawk)
        2.  Local minima, saddle points
          In R^10000,  SADDLE POINTS are very common & a problem.  
          Local minima are not such a big deal (they mean in EVERY dimension I change value, the loss goes up)
      Momentum fixes a lot of these problems
      Even SINGLE SIMPLE momentum term will fix it
      How does momentum fix the "high condition number" problem?

______________________________

    optimization algos
      AdaM is probably the best algo


      Momentum
        Can overshoot sometimes
        Usually a bug, not a feature (steep minima may not be real; they may be due to biases in the dataset b/c training dataset size small)
      AdaGrad
        Slower than momentum
        But may stop too soon
          (Good in convex case, but in nonconvex case it sucks.
      RMSProp fixes the AdaGrad problem of stopping too soon  ()
      AdaM
        Momentum mixed with  RMSProp
        (kinda)
        FIRST STEP IS TOO BIG b/c we're dividing by second moment (we're subtracting by the 2nd moment)

        Johnson gave us the ideal initial setting of AdaM & lr
          beta1=0.9
          beta2=0.999
          lr=1e-3 or 5e-4
        AdaM doesn't fix EVERYTHING
          (see time 44:57 for technical explanation of a problem AdaM DOESN'T solve)

        L-BFGS
          Less stochasticity
          Good if you can afford full-batch GD
          Style transfer
          Less often for training NNs



        around (49:26)
        Second-order version of "lr decay"  can step more quickly to the minimum (right to it!)
          This is a Taylor series   using the 2nd term
          But the quadratic approx isn't perfect
          You can get away with no LR
______________________________

        Model ensembles
          Better than just minimizing training error
          The point is, VALIDATION accuracy is what we care about
______________________________

        Regularization:
          Regularization is better than ensembles?
          Pattern in regularization: 
            1.  Add randomness in during training
            2.  Take out the randomness during test time

          Single models
          Dropout makes more sense for NNs than just L2 norm penalty
            "Mask" that randomly zeroes out particular neurons during training
          1.  "one neuron learns 'furriness'; another learns 'tail-ness;'    etc."
            (less interdependency between neurons/layers where everything learns the same thing)
          2.  "Model ensembling within 1 model"

          But what happens at test time?
            (maybe we should turn off dropout at same epoch N-9)
            OR; we  just get rid of it at test time

          Pattern in regularization: 
            1.  Add randomness in during training
            2.  Take out the randomness during test time

          "Inverse dropout"
______________________________

          Time: 1:03:55

          Batch norm
            Fits the following description of regularization:
            Pattern in regularization: 
              1.  Add randomness in during training
              2.  Take out the randomness during test time
______________________________

          Time: 1:04:48

          Data augmentation
            Horizontal flips
            Crops (cropped cat is still a cat)
            Vary the contrast or brightness

______________________________


          "DropConnect"
          Fractional max-pooling
          Stochastic Depth

          Best to regularize AFTER you see problems   rather than blindly adding them
            Except batchnorm; that's always a good idea.


______________________________


    Transfer learning (1:09:40)
      (ie. retraining a ResNet for Body-Measuring or VGG-16 or ...)
      We never have enough data (use all of Hinton's pre-work on  CIFAR-10)
      Another way 

      Have to reinitialize parts of the network in a thoughtful way 
        (maybe it's mostly the last layer(s) ?? )
        Redo the later layers.

        Remember to freeze the weights (no backprop into 'em) of the earlier layers
        Basically train a linear classifier on top of the last few layers
        Drop the Learning Rate significantly.

      This is less possible for specialized problems where the images differ from ImageNET/CIFAR-100 a lot
        ie. CAT scans

      Transfer learning is EVERYWHERE
        DEFINITELY useful for our cases
        Esp. if people are in the training data (for "Cat")


___________________________________________________________________________________________________

  Lecture  9: CNN architectures
___________________________________________________________________________________________________





  Summary: CNN architectures
    Case studies:
      1.  AlexNet
      2.  VGG
      3.  GoogLeNet
      4.  ResNet
  ___________________________________________________________________________________________________
    




    New research:
      NiN (Network in Network)
      Wide ResNet
      ResNeXT
      Stochastic Depth
      DenseNet
      FractalNet
      SqueezeNet


    AlexNet
      Destroyed previous ML (non-Neural-Network) CVision models
      Ensemble (many AlexNets were trained in parallel and then they did majority-voting)
      Trained on 2 GPUs at once
      8 layers
  ___________________________________________________________________________________________________

    VGGNet
      Oxford

      Much deeper
      16-19 layers
      Small (tiny) filters  (3x3)

      What does "1st place in localization" mean?
        Bounding box



      Did better than GoogLeNet on certain tasks
      3x3 Conv filters *ALL THE WAY* .   (3x3 is the smallest possible filter that has *ANY* spatial awareness.
      But also very deep

  ___________________________________________________________________________________________________


    GoogLeNet
      2014 winner

      ImageNet challenge
        Killed everything but VGGNet
      No Fully Connected layers
        Fewer parameters (fewer even than the much "shallower" AlexNet)
      Inception module

      The original LeNet:
        MNIST hand-written digit recognition  ("OCR")
  ___________________________________________________________________________________________________

    ResNet
      2015 winner ImageNet

      What is a "residual?"




      Different paradigm than previous architectures
        Hugely deeper
      152 layers

      Problems of straight depth AREN'T caused by overfitting
        Deeper models are harder to optimize
        If "learning" were straightforward, the last layers could just learn the identity mapping
        Deeper should work AT LEAST as well as an NN with similar architecture but shallower


      More recent winners:
        http://image-net.org/challenges/LSVRC/2016/results
      2018:
        https://jiankangdeng.github.io

      Skip connection
        Learn F(x) = H(x) - x
        (x == "prev layer's input")
        The idea is that learning the identity mapping would be ideal.  So basically, we cheated and "fed the identity" to the next layer (ie. the skipped-ahead outputs of prev layer's neurons go to the next layer / "next-next layer")
        "What is it that we need to add to the original input to 'learn?'   "
        The new thing we're learning (F(x) in this notation) is the "residual"

      Their theory may not be right, but their model sure works.
        Who knows why?


      No dropout
        Reason??

      ResNet + Inception combo
        How about newer networks?
        Any new techniques?
  ___________________________________________________________________________________________________

    (time: 1:07:38)
    Ideas since (more recent) that improve upon the ResNet performance:
      1. "Identity Mappings in Deep Residual Networks"
      He et al.,  2016
        More direct path for propagating information throughout the network (moves activation to residual mapping pathway)
        Better performance than ResNet

      (time: 1:08:40)
      2. "Wide Residual Networks"
      Zagoruyko et al.,  2016
      Residuals' effect > Depth's effect
        Therefore, wider residual blocks
          Wider are more parallelizable
          Depth is sequential  (~= resistors in series vs. parallel)
        50-layer wide ResNet outperforms 152-layer original ResNet

      3. ResNeXT
      (time == "1:10:56")
      Xie et al. 2016
        This "ResNeXT" paper is by the original author(s) of ResNet
        Width of residual block through multiple parallel pathways ("cardinality")
          See the picture in the slides; it'll be much clearer what this "MULTIPLE PARALLEL PATHWAYS" means

        Similar in spirit to Inception module

      4. "Deep Networks with Stochastic Depth"
      (time == 1:11:21)

      Stochastic Depth
        Motivation: reduce vanishing gradients and training time through short networks during training

        Randomly drop a subset of layers during each training pass
        Dropout but for ResNets
          (Dropout but for skip layers)
        Use the full deep network at test time
        
        Bypass with identity function



  ___________________________________________________________________________________________________

    NiN ("Network in Network")
      Lin et al. 2014
    
    
      1.  "micronetwork" within each conv layer to compute more abstract features for local patches
        a.  computes more abstract features for local patches
      2.  Inspiration for GoogLeNet
      3.  The micronetwork uses MLP (FC, ie. 1x1 conv layers)
      4.  It calculates more complex features to feed into the Neural network
      5.
      6.
      7.
      8.
      9.


  ___________________________________________________________________________________________________

   Personal pet theory:
    Could we do HOG => a CNN?
    Some kind of complex preprocessing of the data to get the performance we REALLY want
    (idea for "Cat")
    Complex, distilled features like edges, outlines, "tails," "blobs of fat around your midsection," but going into the CNN   instead of used raw








  Beyond ResNets:
    1. "FractalNet: Ultra-Deep Neural Networks without Residuals"
      Larsson et al. 2017

      More dropout for regularization
        "regularization" is also kind of like saying "keep the network learning"

      Arguments: 
        The key is transitioning effectively from shallow to deep.
        Residual representations are not necessary
        "Fractal" architecture with both shallow and deep paths to output
        Trained with dropping out sub-paths
        Full network at test time

      There are BOTH shallow and deep paths to the end Loss value
        This is more variable than the ResNet.  I wonder if you could visualize the outputs of each "sub net" as an image and actually understand what each subnet is doing
___________________________________________________________________________________________________

    2. "Densely Connected Convolutional Networks"
    (time == 1:13:58)
      Huang et al. 2017

      nxb:  I've heard of this.  Idk if that means it's 1. good or   2.  just trendy.



      Dense blocks where each layer is connect to every other layer   (feedforward)


      Problems fixed:
        1. Fixes the "vanishing gradient" problem
        2. Encourages feature reuse
___________________________________________________________________________________________________

    3.  SqueezeNet:
      AlexNet-level Accuracy with 50x Fewer Parameters and <0.5Mb Model Size

      Iandola et al. 2017

      Efficient.

      Fire modules consisting of a 'squeeze' layer with 1x1 filters feeding an 'expand' layer with 1x1 and 3x3 filters (see the image; it helps communicate the big idea)
      AlexNet level accuracy on ImageNet with 50x fewer parameters
      Can compress to 510x smaller than AlexNet


      Good for mobile, in-browser NNs





  ___________________________________________________________________________________________________

  Summary: CNN architectures
    Case studies:
      1.  AlexNet
      2.  VGG
      3.  GoogLeNet
      4.  ResNet
    
    New research:
      NiN (Network in Network)
      Wide ResNet
      ResNeXT
      Stochastic Depth
      DenseNet
      FractalNet
      SqueezeNet

  ___________________________________________________________________________________________________

    ResNet widely available, used everywhere

    Trend towards very deep networks
      (except the most recent ResNet)







  https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=9


















  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________

      MIT's more recent CNN lectures:
        https://www.youtube.com/watch?v=H-HVZJ7kGI0
  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________
  ___________________________________________________________________________________________________




___________________________________________________________________________________________________

  cs231n  Lecture 12: Visualizing and Understanding
___________________________________________________________________________________________________

  t-SNE is like PCA but nonlinear  (Deep Learning people use it to reduce dimensionality)
    ie. looks just like that famous MNIST image   on xy Cartesian coordinates
      (https://www.researchgate.net/profile/Efthymios_Tzinis/publication/326572550/figure/fig5/AS:651846066663431@1532423621302/2D-manifold-of-MNIST-dataset-learned-using-t-SNE.png)

  How do we visualize deep layers of CNNs  (video on Jason Yosinski 's  website)   (23:00 and a little before)
    **Very** VERY related to NN interpretability & debugging


  Wait, what does the "activation map" MEAN   in the middle layers of a CNN?
    / how is it mathematically defined?


  There's also KNN on late layers of the network.
    Maps all the types of images from images onto 2 dimensions   (https://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_full_4k_seams.jpg)


  "Maximally Activating Patches" :   (a little before 26:00 in the YouTube video for Lecture 12)
    Shows the types of images   that activate particular Neurons

  Occlusion Experiments: (29:13)
    "Visualizing and Understanding Convolutional Networks" (2014)
    Mask part of elephant 
    Which part makes you less confident it's an elephant?

  Saliency Maps:  (31:30)
    *** Which pixels matter for classification? ***
    More **PRECISE** than  the occlusion mask experiments?
    Maybe just different

    Useful for Semantic segmentation
      Karen's paper?
      No need for labeled data!
        Great!
        But performance SUCKS.
        More of a cool novel idea than a useful 
      Grabcut

  Intermediate Features via (guided) backprop  (34:38)
    Compute the gradient of neuron value w.r.t. image pixels.
    Only backprop positive gradients through ReLUs
    Zeiler and Fergus, "Visualizing and Understanding Convolutional Networks" , ECCV 2014
    Used in tandem with previous technique to verify what particular neurons are "looking for"




  Visualizing CNN features: (38:10 started,   but the SUCCESS is at 40:39  )
    Paper name:
      Yosinski et al. "Understanding Neural Networks through Deep Visualization" 2014
        Random noise is what maximizes the confidence of the Detection ("Detection" is AKA "Identification") score.
    1. Gradient Ascent
    2. Guided backprop
    But **ALSO** we want it to be a nice-looking image
    Find (*synthesize* / make) the image that maximally activates a particular neuron.
    If this technique is  applied this to **ALL** the neurons in the **WHOLE** CNN, maybe we can picture everything each neuron is doing.
      I think this is basically my idea.  Maybe we would like a more "miniature" version of this same idea, though.   (ie. "What are the particular filters that this neuron fires for,   and can we look at only **THOSE** features")  

    Better regularization => better images

    Nguyen et al.,  "Synthesizing the preffered inputs for neurons in neural networks via Deep Generator Networks"
      2016, NIPS
      Optimize in FC6 latent space instead of pixel space


    Fooling image networks (47:43)
      (ie.  Crashing self-driving cars)
      Random-seeming noise
      Incomprehensibly different
      But then why do the NNs still WORK    SO. FUCKING. WELL. on regular images?
        More from Goodfellow   in guest lecture  (GANs)

    Why do we care?
      Criticism of black-box-i-ness    compared to "shallow learning" methods.   than  Neural Networks

    Deep Dream (52:50)
      Amplify existing features
      Does more of whatever a neuron "wants to see"   in an image

    Feature Inversion  (57:10)
      "Give us an image that matches this given feature vector"
      Deeper parts of the network   lose more of the idea of "what does a photo of an elephant look like?"
      1st layer basically preserves the original image

    Texture synthesis (59:59)
      Old old graphics problem  (make a vgame background)
        Classical CV just copies pixels
      Gram Matrix does ("hand-wavily") "which pixel values tend to cooccur with which other pixel values in the (spatially) nearby vicinity?"
        Kind of like covariance
        Computing this Gram matrix is less computationally expensive than   computing a *proper* covariance matrix
      We could use Style Transfer to make video game    textures/backgrounds.
        Computing the gram matrices on Deeper layers  of the network reproduces more of the image at a time


    Neural Style Transfer (1:09:24)
      More control of what the image looks like   than we get over DeepDream
      You can blend multiple style images    & put both of them in your constructed image.
      DeepDream + StyleTransfer
        DogSlugs + Starry Night  + Stanford campus
      Can be slow if done naively.    Have to pre-train another "style network,"   but then you can do fast StyleTransfer
    Fast-style-blended-multiple-styles
      Also a thing you can do.






















































