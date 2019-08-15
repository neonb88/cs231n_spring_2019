from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

from debug import debug
from copy import deepcopy
from collections import OrderedDict # added by nxb, July 10, 2019,        22:42:34 EDT (it's almost 11 P.M. in Delaware)

# nxb's   shorterned nicknames for functions:
drop_fwd=d_fwd=dropout_forward   # from cs231n.layers  (cs231n/layers.py)
drop_bwd=d_bwd=dropout_backward  # from cs231n.layers  (cs231n/layers.py)



#================================================================================================
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
#================================================================================================
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(scale=weight_scale, size =(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        from collections import OrderedDict
        self.debug_info=OrderedDict()
        self.debug_info['initial_W1'] = self.params['W1'].copy()
        self.debug_info['initial_b1'] = self.params['b1'].copy()
        self.debug_info['initial_W2'] = self.params['W2'].copy()
        self.debug_info['initial_b2'] = self.params['b2'].copy()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
#========= end func definition of "__init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):"============


#================================================================================================
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']

        logits_layer_1, cache1  = affine_forward(X, W1, b1)
        ReLUed, cache_ReLU      = relu_forward(logits_layer_1) # np.array_equal(cache_ReLU, logits_layer_1)   (these 2 quantities are equal) -nxb, July 9, 2019,  ~12:00 A.M. (midnight)
        logits_layer_2, cache2  = affine_forward(ReLUed, W2, b2)
        scores                  = logits_layer_2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Losses:
        reg             = self.reg
        loss, dscores   = softmax_loss(logits_layer_2, y)
        loss           += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))      
        # why don't we regularize the biases??? -nxb, July 9, 2019,   1 A.M. (technically it's July 10, 2019, 1 A.M.)

        # Derivatives:
        dX2, dW2, db2   = affine_backward(dscores,   cache2  )  # should "logits_layer_2" and/or "ReLUed" be in here?
        dReLU           = relu_backward(dX2, cache_ReLU)  #???? dW2?   but doesn't db2 go in there somewhere??   or is it dscores?
        dX1, dW1, db1   = affine_backward(dReLU, cache1)
        grads['W1']     = dW1 +(reg* W1)
        grads['b1']     = db1     
        grads['W2']     = dW2 +(reg* W2)
        grads['b2']     = db2
        # why don't we regularize the biases??? -nxb, July 9, 2019,   1 A.M. (technically it's July 10, 2019, 1 A.M.)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
#========================end func definition of  "loss(self, X, y=None):"===========================

#===================================================================================================
class FullyConnectedNet(object): # <== buffer on left is within   class FCNet.
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

#===================================================================================================
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        if debug:
          print("self.use_dropout:",self.use_dropout)
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)   
        #*****    n_layers = 1 + len(hidden_dims)    (b/c hidden_dims is a LIST of ints, not a single integer) -nxb   and his poor reading, and his impatience
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        n_layers=self.num_layers
        n_classes=num_classes
        # trivial/degenerate NN (I think the single-layer perceptron   is equivalent to the multinomial logistic regression (ie. SVM/Softmax)  )
        if n_layers==1:
          # TODO TODO TODO:   this block of code is probably trahs.  I haven't updated it since I irst wrote it.  (in my defense, who uses a "Neural Network" with 1 layer anyways?  Hardly an NN at all)  -nxb,  July 18, 2019
          # TODO TODO TODO
          # TODO TODO TODO
          # TODO TODO TODO
          # TODO TODO TODO
          self.params['W1']= weight_scale * np.random.randn(input_dim, n_classes)
          self.params['b1'] = np.zeros(n_classes) #  TODO:    do I need to call .reshape() to (n_classes,1) or (1,n_classes)?

          # "self.debug_info" is useful during   hyperparameter cross-validation, among other places:   how do we understand which NN did well & why   after we find a high-performer?  -nxb, July 10, 2019
          self.debug_info=OrderedDict()
          self.debug_info['W1']= self.params['W1'].copy()
          self.debug_info['b1']= self.params['b1'].copy()

          #   There's probably a more general (ie. BETTER) way of doing this, but let's iterate towards perfection;   
          # the main goal here is to learn NNs, not to perfect this academic (purely for the purposes of teaching) codebase
          # -nxb, July 10, 2019

          # Looking back at this code on July 15, 2019, I think maybe the code I ended up writing for "else: (n_layers > 0)" actually IS general enough.  TODO: double-check after we implement batch_norm().

          # TODO TODO TODO
          # TODO TODO TODO:      batch norm
          # TODO TODO TODO        generalize the "if" branch into an "if-else"  (refactor out this "if" branch)

        else:
        #===================================================================================================
        # n_layers > 0
        #===================================================================================================
          prev_n_neurons=input_dim
            # 
            # TODO: in for loop,      deal with batch_norm, 
            # TODO:                             dropout, and 
            # TODO:                             layer_norm
            # TODO:                             layer_norm
          # not "in range(n_layers)" becasue that would be too many; at the end we need to do "n_classes."
          for layer_idx in range( len(hidden_dims)   ):

            # weights:
            self.params['W'+str(layer_idx+1)] =\
              weight_scale *\
                np.random.randn(
                  prev_n_neurons, 
                  hidden_dims[layer_idx]
                ).astype('float64')
            # biases:
            self.params['b'+str(layer_idx+1)] = np.zeros(hidden_dims[layer_idx] ,).astype('float64')

            if self.normalization: # default val of self.norm... is "None"
              # Batch Normalization:
              if 'batchnorm' == self.normalization.lower():
                # gamma == 'g':
                self.params['g'+str(layer_idx+1)]   = np.array(1.).astype('float64')
                # 'beta' because 'b' was already taken by 'bias.'   -nxb, July 14, 2019
                self.params['beta'+str(layer_idx+1)]= np.array(0.).astype('float64')  # I do these "np.arraY()" calls for the stupid hacky reason of  the later line "self.params[k] = v.astype(dtype)"    b/c
            # end block "if self.normalization:"

            prev_n_neurons = hidden_dims[ layer_idx]
          #========= end for loop ========= 

          # final weights:
          self.params['W'+str(n_layers)] =\
            weight_scale *\
              np.random.randn(
                prev_n_neurons, 
                n_classes
              ).astype('float64')
          # final biases:
          self.params['b'+str(n_layers)] = np.zeros(n_classes ,).astype('float64')

          # No batch norm layer before the output softmax   (see their API specification:
          #   "{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax"
          #)




          #   There's probably a more general (ie. BETTER) way of doing this initialization (ie. with the last layer inside the for loop   & no conditional w/in the loop), but let's iterate towards perfection;   
          # the main goal here is to learn NNs, not to perfect this academic (purely for the purposes of teaching) codebase
          # -nxb, July 10, 2019

          # "self.debug_info" is useful during   hyperparameter cross-validation, among other places:   how do we understand which NN did well & why   after we find a high-performer?  -nxb, July 10, 2019
          self.debug_info=OrderedDict(  deepcopy(self.params)  )
        # end else: (if n_layers > 0)
        #===================================================================================================


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
#====================================== end func definition of " __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10, dropout=1, normalization=None, reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None):"====================================


#=================================================================================================================================
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Make it easy to reference these variables in a smaller number of characters: 

        n_layers=self.num_layers
        X_in=X
        X=X_in.copy() # deepcopy()
        caches_affine=[]         #NOTE TODO NOTE:   rename cache
        caches_ReLU=[]
        caches_bnorm=[]
        caches_dropout=[]

        loss=0
        reg=self.reg
        bn_param={'mode':     'train',
                  'eps' :     1e-5,
                  'momentum':  0.9}

        # NOTE:   improvement:   why didn't I just call EVERYTHING "X" instead of having names like "X_curr" and "logits" flying around   for no reason?   -nxb, July 15, 2019
        # the REAL question is:  will this cosmetic change affect the EFFECT of the  code in any way?

        #=======================================================================================
        # Forward Pass:
        #=======================================================================================
        for i in range(n_layers - 1):
          # the "- 1" is because at the end we do softmax, not another activation func (ie. ReLU)

          # W and b are 1-indexed,  not 0-indexed:
          W=self.params['W'+str(i+1)].copy()
          b=self.params['b'+str(i+1)].copy()

          # Affine:
          X, cache_affine=affine_forward(X, W, b)
          caches_affine.append(cache_affine)

          # Normalization:
          if self.normalization:
            if self.normalization == 'batchnorm':
              X, cache_bnorm = bnorm_fwd(X, self.params['g'+str(i+1)], self.params['beta'+str(i+1)],  bn_param, which_layer=i+1)
              caches_bnorm.append(cache_bnorm)

            elif self.normalization =='layernorm':
              pass  # TODO:  implement layernorm.   But apparently they didn't have us implement this.  So woohoo!   yaaaaaaaaaaaaaaaay!  also: wut.    (is "wut?" preferable to "wut."  ?)
          #========================= end block   "if self.normalization:"=========================

          # ReLU  (nonlinearity)
          X, cache_ReLU=relu_forward(X)
          caches_ReLU.append(cache_ReLU)
          #=======================================================================================
          #==================================== dropout_fwd ======================================
          #=======================================================================================
          #=======================================  TODO  ========================================
          #=======================================  TODO  ========================================
          #=======================================  TODO  ========================================
          if self.use_dropout:
            X, cache_dropout=drop_fwd(X, self.dropout_param)    # forward
            caches_dropout.append(cache_dropout)
          #=======================================  TODO  ========================================
          #=======================================  TODO  ========================================
          #=======================================  TODO  ========================================
          #=======================================================================================
          #=======================================================================================
          #=======================================================================================

          # Regularization Loss
          reg_loss=np.sum(W*W)
          loss+=  reg_loss
        loss*= 0.5*reg

        # Final scores:
        W_last=self.params['W'+str(n_layers)].copy()
        b_last=self.params['b'+str(n_layers)].copy() # 30
        scores, cache_affine=affine_forward(X, W_last, b_last)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # The NN architecture:
        #   "  {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) 
        #     - affine - softmax"

        ############################################################################
        # bwd pass  (backpropagating the partial derivatives)
        ############################################################################
        if debug:
          print('\n'*2 + '='*99)
          print("starting  backward pass...")
          print('='*99 + '\n'*2)

        # softmax instead of ReLU this time:
        data_loss, dsoftmax = softmax_loss(scores, y) # scores.shape == (N, C)  (N_minibatch, num_classes)
        loss  += data_loss # we already did the regularization_loss

        dX, dW, db = affine_backward(dsoftmax, cache_affine)
        grads['W'+str(n_layers)]= dW
        grads['b'+str(n_layers)]= db

        if debug:
          print("after 1st affine_backward pass, dX = {0}".format(dX[:3,:3]))

        # the value of "i" goes from n_layers-2   to 0    (intuitive, traditional C-based/Java-based 0-indexing, not 1-indexing.   Only for the weight names do we use the human-friendly 1-indexing (ie. 'W1', 'W2', etc.))  -nxb
        if debug:
          print("self.use_dropout = {0}".format(self.use_dropout) )

        # Main backprop   for loop:
        for i in range(n_layers - 2, -1, -1):

          # drelu -> dbatchnorm -> daffine
          #   The order is reversed from the forward pas s

          # Dropout:
          if self.use_dropout:
            dX =  drop_bwd(dX, caches_dropout[i])
            if debug:
              print("after  dropout, dX = {0}".format(dX[:3,:3]))
            #assert dX  # was gonna do   something like   "assert dX_old.isclose(dX_new * p)"  (but we'd have to vectorize "math.isclose()"   )  # or / p,   ("np.count_nonzero(math.isclose(  dX_old, dX_new ))"   )  or 
            #grads['d%i'%(i+1)]=dX   # we don't update dropout;  the 'p' remains the same all the time

          # ReLU
          dX = relu_backward(dX, caches_ReLU[i])
          if debug:
            print("after  ReLU, dReLU = {0}".format(dX[:3,:3]))

          # Batch/layer normalization:
          if self.normalization:
            if self.normalization.lower() == 'batchnorm':
              dX, dgamma, dbeta = bnorm_bwd(dX, caches_bnorm[i])         #caches[])  TODO NOTE TODO:       verify which kind of cache goes in here.  some kind of          
              if debug:
                print("after  batchnorm, dX = {0}".format(dX[:3,:3]))
                print("dg{0} =  {1}".format(i+1, dgamma))
                print("dbeta{0} =  {1}".format(i+1, dbeta))
              grads['g'+str(i+1)]=dgamma
              grads['beta'+str(i+1)]=dbeta

          # FC / affine
          dX, dW, db = affine_backward(dX, caches_affine[i])  #dReLU, caches_affine[i])
          if debug:
            print("after  affine_bwd, dX = {0}".format(dX[:3,:3]))
          W=self.params['W'+str(i+1)]
          grads['W'+str(i+1)]=dW + (reg*W)
          grads['b'+str(i+1)]=db
                                                                                                                                                                                                      

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
#=========================================end func definition of " loss(self, X, y=None):"========================================


































































































#===================================================================================================
def bnorm_fwd(x, gamma, beta, bn_param, which_layer):
    """
    NOTE:  this function works by side effects:  "bn_param" gets mutated throughout the course of this function's execution, and that's how the running averages are updated for the NEXT time "bnorm_fwd()" is called.  This is suboptimal design, but it USUALLY works, and I'm too lazy to change it at this moment.  Definitely should be changed, though.  -nxb, July 15, 2019
      I think I should do these things more when I have to maintain the code later (ie. startup/PRODuction code)


    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean  = momentum * running_mean + (1 - momentum) * sample_mean
    running_var   = momentum * running_var  + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean  = bn_param.get('running_mean'+str(which_layer), np.zeros(D, dtype=x.dtype))
    running_var   = bn_param.get('running_var '+str(which_layer), np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':

        # Current forward pass calculation:
        DOWN=0
        sample_mean = mean=np.mean(x, axis=DOWN, keepdims=True)
        sample_var  = var = x.var(axis=DOWN)  # this line is equivalent to mine, but  is much simpler to read.  Idk about timing; I would care if I absolutely  needed to optimize for speed.

        '''
        assert np.allclose(var, 
          (np.sum(
            np.square(x-mean), axis=DOWN, keepdims=True)   # nxb:     the old line was var = np.sum(...
            / N)
        )

        assertion passed
        '''

        x_hat= (
          (x-mean)
          / np.sqrt(var + eps))
        out= ( gamma* x_hat) + beta

        #p rint("sample_mean.shape: ", sample_mean.shape) # (1,5)  (1,D)
        #p rint(   (x-mean).shape) # (4,5)
        #p rint(var.shape) # (1,5) == (1,D),   just as it SHOULD be.
        #p rint(x_hat.shape) # (4,5),   just as it SHOULD be.  
        #p rint(out.shape) # (4,5) == (N,D)

        # Storing for later:
        # TODO: make sure cache is right -nxb, July 13, 2019
        cache=(x, gamma, beta, x_hat, mean, var, eps)
        running_mean = (momentum * running_mean) + (1 - momentum) * sample_mean
        running_var  = (momentum * running_var ) + (1 - momentum) * sample_var

        #print(running_mean.shape) # (1,5) == (1,D)    #print(running_var.shape) # (1,5) == (1,D)
    elif mode == 'test':
        x_hat = (x - running_mean ) / np.sqrt(running_var + eps)
        out   = (gamma * x_hat) + beta
        # "cache=None" because we don't need to store partial derivatives for backpropping during test time:
        cache=None

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'+str(which_layer)] = running_mean
    bn_param['running_var'+str(which_layer)] = running_var

    return out, cache
#================== end func definition of  "bnorm_fwd(X, gamma, beta, bn_param):"==================













































































































#===================================================================================================
def bnorm_bwd(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    This is the Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)

    """
    dx, dgamma, dbeta = None, None, None

    x, gamma, beta, x_hat, mean, var, eps = cache
    mu=mean
    m=x.shape[0]
    DOWN=0

    dgamma  = np.sum( (dout * x_hat), axis=DOWN, keepdims=True)
    dbeta   = np.sum(dout, axis=DOWN, keepdims=True)

    # src:   https://kevinzakka.github.io/2016/09/14/batch_normalization/
    inv_var = 1. / np.sqrt(var+eps)
    N, D    = dout.shape
    dxhat   = dout * gamma
    dx      = inv_var / N * (N*dxhat - np.sum(dxhat, axis=0, keepdims=True) 
      - x_hat * np.sum(dxhat*x_hat, axis =0 ))
    ve=var+eps
    #dx      =dout*gamma* (((-x*x+4*x*mu-3*mu*mu)/N/ve) +N+1)/(ve*0.5)

    return dx, dgamma, dbeta
#================== end func definition of  batchnorm_backward_alt(dout, cache):==================
#bnorm_bwd=batchnorm_backward_alt  # these no longer work b/c from 2018 to 2019 they changed the code s.t. FCNet() can't use the same code as a TwoLayerNet()
#bnorm_fwd=batchnorm_forward


# My "bnorm_bwd()" function LOOKS the same as Xu's as far as I can tell.  I think the difference is in the order of the batch_norm/dropout;  ;
























































###########################################################################
###########################################################################
#   This version of " def batchnorm_backward_alt(dout, cache):  " is from
#     https://raw.githubusercontent.com/haofeixu/stanford-cs231n-2018/master/assignment2/cs231n/layers.py
#   this code is from Haofei Xu:   (https://raw.githubusercontent.com/haofeixu/stanford-cs231n-2018/master/assignment2/cs231n/layers.py)
###########################################################################
###########################################################################



#===================================================================================================
def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
      
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """


    ###########################################################################
    ###########################################################################
    # from https://raw.githubusercontent.com/haofeixu/stanford-cs231n-2018/master/assignment2/cs231n/layers.py
    #   this code is from Haofei Xu:   (https://raw.githubusercontent.com/haofeixu/stanford-cs231n-2018/master/assignment2/cs231n/layers.py)
    ###########################################################################
    ###########################################################################

    # The problem is NOT in my version of "bnorm_bwd_alt()."  I copied all the variables over from Haofei Xu's version of this function, and I got the same error.  So maybe it's...  IDK!  I was gonna guess the problem is something to do with the gradient check???  - July 19, 2019 

    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    N, D = dout.shape
    
    #xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache          # <====   Original line from Xu
    x, gamma, beta, x_hat, mean, var, eps = cache               # <====  nxb's edits b/c we set up our fwd() passes differently (they cache different things)
    xhat=x_hat
    ivar = 1. / np.sqrt(var+eps)

    dxhat = dout * gamma
    
    # Ref: https://kevinzakka.github.io/2016/09/14/batch_normalization/
    dx = 1.0/N * ivar * (N*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat*dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

#bnorm_bwd=batchnorm_backward_alt



































































































