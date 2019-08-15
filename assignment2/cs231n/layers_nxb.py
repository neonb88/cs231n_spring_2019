from builtins import range
import numpy as np

# nxb's imports:
from copy import deepcopy
import math
from debug import debug










#==============================================================================
def prn_debug():
    '''
      Just a sample debug statement (prints only once instead of a million times within a loop)
    '''
    fname="/home/cat_macys_vr/.debug/print_once.py" 

    with open(fname, "r") as f:
      txt= f.read()
      go = "False" in txt
    if go:
      var_names = ('x', 'gamma', 'beta', 'x_hat', 'mean', 'var', 'eps')
      for i,val in enumerate(cache):
        print(var_names[i])
        if 'shape' in dir(val):
          print(val.shape)

      # I do this to make  it   only print once
      with open(fname, "w") as f:
        f.write("True")










#==============================================================================
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    dim_2  = np.prod(x.shape[1:])

    out = (x.reshape((x.shape[0], dim_2))   # x.shape from (2,4,5,6)  to  (2,120)
      ).dot(w)\
      +b   

    # example dimensions:
    # x.shape==(  2, 120)
    # w.shape==(120,   3)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache
#================== end func definition of  affine_forward(x, w, b):==================


#==============================================================================
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # partial derivative calculations from [here]( http://cs231n.github.io/neural-networks-case-study/)

    dw=((x.reshape((x.shape[0],
      np.prod(x.shape[1:])))
    ).T).dot(
      dout)
    db=np.sum(dout, axis=0, keepdims=True)
    dx=((  w.dot(
      dout.T)  
    ).T).reshape(
      x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
#================== end func definition of  affine_backward(dout, cache):==================


#==============================================================================
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out=np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache
#================== end func definition of  relu_forward(x):==================


#==============================================================================
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #   (the cache from the previous layer's forward pass  through ReLU has to have not been zeroed out)
    #   (the activations from the previous layer's forward pass through ReLU  (stored in the "cache" variable)  have to have not been zeroed out)

    # What we're doing:
    # Let previous activations through   iff the particular element of x>0.
    dx  = dout.copy()
    dx[x <= 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
#================== end func definition of  relu_backward(dout, cache):==================


#==============================================================================
def batchnorm_forward(x, gamma, beta, bn_param): # buffer over there is also in bnorm_fwd()>
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

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
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Current forward pass calculation:
        DOWN        = 0
        sample_mean = mean  = np.mean(x, axis=DOWN, keepdims=True)
        sample_var  = var   = (np.sum( np.square(x-mean), axis=DOWN, keepdims=True)    / N)
        x_hat= ( (x-mean)   / np.sqrt(var + eps))
        out= (gamma * x_hat) + beta
        # I could have just used np.var()  -nxb

        # Storing for later:
        cache=(x, gamma, beta, x_hat, mean, var, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var  = momentum * running_var  + (1 - momentum) * sample_var

        if debug:
          pass
          #p rint  ("sample_mean.shape: ", sample_mean.shape) # (1,5)  (1,D)
          #p rint  (   (x-mean).shape) # (4,5)
          #p rint  (var.shape) # (1,5) == (1,D),   just as it SHOULD be.
          #p rint  (x_hat.shape) # (4,5),   just as it SHOULD be.  
          #p rint  (out.shape) # (4,5) == (N,D)
          #p rint  (running_mean.shape) # (1,5) == (1,D)
          #p rint  (running_var.shape) # (1,5) == (1,D)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out   = (gamma * x_hat) + beta
        # "cache=None" because we don't need to store partial derivatives for backpropping during test time:
        cache=None

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache
#================== end func definition of  batchnorm_forward(x, gamma, beta, bn_param):==================


#==============================================================================
def batchnorm_backward(dout, cache): # bottom buffer
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, gamma, beta, x_hat, mean, var, eps = cache
    #p rint  (eps) # float (no eps.shape)
    #p rint  (var) # 109.49  (float; no var.shape)
    #p rint  (mean)# 12.53976... (float; no mean.shape)

    # calculation for dx
    m       = x.shape[0]                                            # dout.shape == (4,5) == (N, D)
    DOWN    = 0
    dx_hat  = dout*gamma              # these dimensions check out.  (dx_hat)  :)
    dvar    = (np.sum( dx_hat * (x - mean), axis=DOWN, keepdims=True) *
      (-0.5 * ((var + eps)**-1.5)))                         # p rint  ("dvar.shape: ", dvar.shape) # (1,5) == (1,D)
    dmean   = (np.sum(  (dx_hat  / -np.sqrt(var + eps)), axis=DOWN, keepdims=True)
      + (dvar * (-2)* np.sum(x-mean, axis=DOWN, keepdims=True) / m)
    )
    dx = ((dx_hat / np.sqrt(var+ eps))
      + (dvar * 2*(x-mean) / m)
      + (dmean / m)
    )

    #dgamma and dbeta are much simpler:
    dgamma  = np.sum( (dout * x_hat), axis=DOWN, keepdims=True)
    dbeta   = np.sum( dout          , axis=DOWN, keepdims=True)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
#================== end func definition of  batchnorm_backward(dout, cache):==================


#==============================================================================
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
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # This calculation is correct!   So whatever problems we were encountering earlier with bnorm_bwd() were completely outside the scope of this function   (I think we were mainly just failing to store enough  values fo gamma and beta in some __init__() function )
    x, gamma, beta, x_hat, mean, var, eps = cache
    mu= mean
    m = x.shape[0]
    DOWN=0
    #===================================================================================================
    dgamma  = np.sum( (dout * x_hat), axis=DOWN, keepdims=True)
    dbeta   = np.sum(dout, axis=DOWN, keepdims=True)

    # src:   https://kevinzakka.github.io/2016/09/14/batch_normalization/
    inv_var = 1. / np.sqrt(var+eps)
    N, D    = dout.shape
    dx_hat   = dout * gamma
    dx      = inv_var / N * (N*dx_hat - np.sum(dx_hat, axis=0, keepdims=True)
      - x_hat * np.sum(dx_hat*x_hat, axis =0 ))
    ve=var+eps
    #dx      =dout*gamma* (((-x*x+4*x*mu-3*mu*mu)/N/ve) +N+1)/(ve*0.5)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
#================== end func definition of  batchnorm_backward_alt(dout, cache):==================


#==============================================================================
def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    # the "HINT" they gave  is really just "1. transpose your data   and 2. use the code for 'batchnorm_fwd()'   "



    #===================================================================================================
    #   fwd :   ("layernorm_forward")
    #===================================================================================================
    # All this code is from  https://github.com/haofeixu/stanford-cs231n-2018/blob/master/assignment2/cs231n/layers.py
    # Tranpose x to use batch normalization code, now x is of shape (D, N)
    x = x.T

	# Just copy from batch normalization code
    mu = np.mean(x, axis=0)

    xmu = x - mu
    sq = xmu ** 2
    var = np.var(x, axis=0)

    sqrtvar = np.sqrt(var + eps)
    ivar = 1./sqrtvar
    xhat = xmu * ivar

    # Transpose back, now shape of xhat (N, D)
    xhat = xhat.T

    # Just copy from batch normalization code
    out = gamma * xhat + beta

    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)




    # My first attempt:
    '''
    # Save mean and variance computations by only calculating once and then just passing values around.
    N,D   =  x.shape
    mu    =  x.mean()
    var   = (x*x).mean() - mu*mu  # shortcut:   is this faster than traditional variance calculation?
    x_hat = (x - mu )  / np.sqrt(var + eps )
    out   = (x_hat*gamma)  + beta

    # Store for bwd pass later:
    cache=(x, gamma, beta, x_hat, mu, var, eps)
    '''



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache
#================== end func definition of  layernorm_forward(x, gamma, beta, ln_param):==================


#==============================================================================
def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****




    # Just copy code from batch normalization backward:    (maybe I could have literally called the function batchnorm_bwd() after mild modifications)
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat*dout, axis=0)

    dxhat = dout * gamma

    # Transpose dxhat and xhat back
    dxhat = dxhat.T
    xhat = xhat.T

    # Actually xhat's shape is (D, N), we use notation (N, D) to let us copy
    # batch normalization backward code when computing dx without change anything
    N, D = xhat.shape

    # Copy from batch normalization backward code
    dx = 1.0/N * ivar * (N*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))

    # Transpose dx back
    dx = dx.T





    '''
    x, gamma, beta, x_hat, mu, var, eps = cache
    N, D    = dout.shape
    DOWN=0

    # dgamma and dbeta:
    dgamma  = np.sum( (dout * x_hat), axis=DOWN, keepdims=True)
    dbeta   = np.sum(dout, axis=DOWN, keepdims=True)


    # calculation for dx
    m       = x.shape[0]                                            # dout.shape == (4,5) == (N, D)
    DOWN    = 0
    dx_hat  = dout*gamma              # these dimensions check out.  (dx_hat)  :)
    '''
    '''
    dvar    = (np.sum( dx_hat * (x - mean), axis=DOWN, keepdims=True) *
      (-0.5 * ((var + eps)**-1.5)))                         # p rint  ("dvar.shape: ", dvar.shape) # (1,5) == (1,D)
    dmean   = (np.sum(  (dx_hat  / -np.sqrt(var + eps)), axis=DOWN, keepdims=True)
      + (dvar * (-2)* np.sum(x-mean, axis=DOWN, keepdims=True) / m)
    )
    dx = ((dx_hat / np.sqrt(var+ eps))
      + (dvar * 2*(x-mean) / m)
      + (dmean / m)
    )
    '''

    '''
    ivar = 1. / np.sqrt(var+eps)
    #ivar=ivar.T
    dx_hat = dx_hat.T
    x_hat = x_hat.T

    # Actually x_hat's shape is (D, N);  we use notation (N, D) to let us copy
    # batch normalization backward code when computing dx without change anything
    N, D = x_hat.shape

    # Copy from batch normalization backward code
    dx = 1.0/N * ivar * (N*dx_hat - np.sum(dx_hat, axis=0) - x_hat*np.sum(dx_hat*x_hat, axis=0))

    # Transpose dx back
    dx = dx.T
    '''


    '''
    # dx:  (1st draft copied directly from batchnorm_bwd_alt()   )
    inv_var = 1. / np.sqrt(var+eps)
    dx_hat   = dout * gamma
    dx      = inv_var / N * (N*dx_hat - np.sum(dx_hat, axis=0, keepdims=True) 
      - x_hat * np.sum(dx_hat*x_hat, axis =DOWN ))
    '''





    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
#================== end func definition of  layernorm_backward(dout, cache):==================


#==============================================================================
def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        '''
        - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
          mask that was used to multiply the input; in test mode, mask is None.
        '''
        mask = (np.random.rand(*x.shape) < p) / p # p is probability of keeping;  np.random.rand() < 1   would mean "keep all the neurons"
        out=x*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out=x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache
#================== end func definition of  dropout_forward(x, dropout_param):==================


#==============================================================================
def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout* mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx
#================== end func definition of  dropout_backward(dout, cache):==================


#==============================================================================
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Convenience names:   (all defined in the docstring for this function definition) 
    N,  Cx, H,  W   = x.shape
    F,  Cw, HH, WW  = w.shape
    assert Cx == Cw
    C=Cx
    stride, pad  = conv_param['stride'], conv_param['pad']

    # Zero padding:
    p = padded  = np.pad( x, [(0,0), (0,0), (pad, pad),  (pad, pad)],  mode='constant',  constant_values=0)

    # We COULD get rid of the following "assert" statements, but  generally the user should feed in this type of input; (they may miscalculated by accident and be grateful for the reminder)
    assert (p.shape[2]-HH) % stride   == 0
    assert (p.shape[3]-WW) % stride   == 0

    # Initialize output "out" with the right size
    H_prime = int( ((H + 2*pad - HH) / stride) + 1)
    W_prime = int( ((W + 2*pad - WW) / stride) + 1)
    out=np.zeros((N,F,H_prime, W_prime), dtype='float64')
    x0=x # Here i keep a pointer to the original "x" so we can reuse the variable name "x" as in high school algebra class' Cartesian x,y,z.  -nxb, July 27, 2019



    #==============================================================================
    # This for loop is obscenely slow at scale:
    for n in range(N):
      for f in range(F):
        for x in range(0, p.shape[3] - WW + 1, stride):
          out_x=int(x/stride)
          for y in range(0, p.shape[2] - HH + 1, stride):
            out_y=int(y/stride)

            # NOTE:   should we `assert` that the Convolution will end PRECISELY where at the END of the padded image,  where it ought to?

            #==============================================================================
            #==============================================================================
            # The creme de la creme:
            #   (The actual real CNN math we're doing)
            #==============================================================================
            out[n, f, out_y, out_x] = np.sum(w[f,:,:,:] * p[n, :, y:y+HH, x:x+WW] )   + b[f]
            #==============================================================================
    #==============================================================================

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    x=x0
    ###########################################################################
    cache = (x, w, b, conv_param)     # This line of code is in "conv_fwd" .
    return out, cache
#================== end func definition of  conv_forward_naive(x, w, b, conv_param):==================


#==============================================================================
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,w,b,conv_param = cache
    N,  _ , H,  W   = x.shape
    F,  C , HH, WW  = w.shape
    stride, pad  = conv_param['stride'], conv_param['pad']
    p = padded  = np.pad( x, [(0,0), (0,0), (pad, pad),  (pad, pad)],  mode='constant' )
    #x0=x    # Here I keep a pointer to the original "x" so we can reuse the variable name "x" as in high school algebra class' Cartesian x,y,z.  -nxb, July 27, 2019

    dx=np.zeros(x.shape)
    dp=np.zeros(p.shape)  # should be almost identical to dx but padded?  Let's test...   No, it's not that simple... -nxb, July 28, 2019
    dw=np.zeros(w.shape)
    db=np.zeros(b.shape)

    for f in range(F):
      # bias:  (db):
      db[f]=np.sum(dout[:,f,:,:])

      for n in range(N):
        for x in range(0, p.shape[3] - WW + 1, stride):
          out_x=int(x/stride)
          for y in range(0, p.shape[2] - HH + 1, stride):
            out_y=int(y/stride)
            assert dw[f,:,:,:].shape ==  dp[n,:,y:y+HH, x:x+WW].shape
            dw[f,:,:,:] +=  dout[n, f, out_y, out_x] * p[n,:,y:y+HH, x:x+WW]   # scalar * values_in_the_shape_of_the_Conv_filter  # success rice!
            dp[n,:,y:y+HH, x:x+WW] += dout[n, f, out_y, out_x] * w[f,:,:,:]

            #=========================================================================#
            # from conv_fwd():
            #out[n, f, out_y, out_x] = np.sum(w[f,:,:,:] * p[n, :, y:y+HH, x:x+WW] )   + b[f]
            #=========================================================================#
    dx=dp[:,:,pad:-pad, pad:-pad] # none of the pad

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
#================== end func definition of  conv_backward_naive(dout, cache):==================


#==============================================================================
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    Hp = int( ((H - HH) / stride) + 1)    # the "p" in "Hp" stands for "prime"
    Wp = int( ((W - WW) / stride) + 1)    # the "p" in "Wp" stands for "prime"
    out=np.zeros((N,C,Hp, Wp), dtype='float64')
    for n in range(N):
      for c in range(C):
        for h in range(0, H-HH+1, stride):
          hout=int(h/stride)
          for w in range(0, W-WW+1, stride):
            wout=int(w/stride)
            out[n,c,hout,wout] = np.max(x[n,c,h:h+HH,w:w+WW])   # pool_fwd

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache
#================== end func definition of  max_pool_forward_naive(x, pool_param):==================


#==============================================================================
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    dx=np.zeros_like(x)
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape

    for n in range(N):
      for c in range(C):
        for h in range(0, H-HH+1, stride):
          hout=int(h/stride)
          for w in range(0, W-WW+1, stride):
            wout=int(w/stride)
            x_slice= x[n,c,h:h+HH,w:w+WW]

            # see https://stackoverflow.com/questions/9482550/argmax-of-numpy-array-returning-non-flat-indices    for details on "unrvael_index"   -nxb, July 28, 2019
            hmax, wmax = np.unravel_index(   np.argmax(x_slice),   (HH,WW)  ) 

            #out[n,c,hout,wout]=np.max(x[n,c,h:h+HH,w:w+WW])  # <==this line of code is  from pool_fwd
            dx[n,c,h+hmax, w+wmax] = dout[n,c,hout,wout]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
#================== end func definition of  max_pool_backward_naive(dout, cache):==================


#==============================================================================
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #===================================================================================================
    #      {       SUBSTANCE (the lines after this are just debugging "print" statements)
    #                       "spatial_batchnorm_forward"
    #===================================================================================================
    bn_param['eps']=1e-5
    (N, C, H, W)  = x.shape
    # usage:   np.transpose((1,0,2,3))  swaps the 1st 2 dimensions   of the 4-D array (AKA rank-4-tensor)
    x   = x.transpose((0,2,3,1)    ).reshape((N*H*W, C))
    #print("Before batchnorm_forward, x.shape == ", x.shape)   # Yeah, x.shape doesn't change HERE.  So why does it change in spatial_groupnorm_fwd()?
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)   # out.shape == (40,3)
    #print("AFTER batchnorm_forward, x.shape == ", x.shape)
    out = out.reshape((N,H,W,C)).transpose((0,3,1,2))
    #===================================================================================================
    #      }   END SUBSTANCE (the lines after this are just debugging "print" statements)
    #===================================================================================================
































    #===================================================================================================
    # It turns out there's no need to reshape these terms after all.  I don't *realllly* know why this is the case, but basically we want them to be spatially oriented (rank 4 tensors rather than rank 2)
    #Xi=0
    #XHATi=3
    # NOTE: I really SHOULD have used a dictionary/hash table instead of this janky-ass list.
    if debug:
      print(cache[Xi].shape)
      print(cache[XHATi].shape)
    #if cache:  # "None" check
      #cache=list(cache)
    #cache[Xi]   = cache[Xi].reshape((N,H,W,C)).transpose((0,3,1,2))
    #print("in fwd()")
    #print( cache[Xi].shape)
    #cache[XHATi]= cache[XHATi].reshape((N,H,W,C)).transpose((0,3,1,2))
    #===================================================================================================

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache
#================== end func definition of  spatial_batchnorm_forward(x, gamma, beta, bn_param):==================


#==============================================================================
def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #===================================================================================================

    # We might have to reshape / transpose some of the contents of "cache"   as well.
    (N, C, H, W)  = dout.shape        # N,H,W,C    # (2,3,4,5)
    #dout = dout.reshape(dout)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.transpose((0,2,3,1)).reshape((N*H*W, C)), cache)
    dx = dx.reshape((N,H,W,C)).transpose((0,3,1,2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
#================== end func definition of  spatial_batchnorm_backward(dout, cache):==================



#==============================================================================
def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****









    # All this code is from  https://github.com/haofeixu/stanford-cs231n-2018/blob/master/assignment2/cs231n/layers.py
    N, C, H, W = x.shape
    x = np.reshape(x, (N*G, C//G*H*W))

    # Transpose x to use batchnorm code
    x = x.T

    # Just copy from batch normalization code
    mu = np.mean(x, axis=0)

    xmu = x - mu
    sq = xmu ** 2
    var = np.var(x, axis=0)

    sqrtvar = np.sqrt(var + eps)
    ivar = 1./sqrtvar
    # the point is they do per-channel   gamma & beta,   but take the mean mu   and std theta over the group as in the nice visualization from the paper.
    #print("mu.shape: ", mu.shape)       # (4,)
    #print("ivar.shape: ", ivar.shape)   # (4,)
    xhat = xmu * ivar
 
    # Transform xhat and reshape
    xhat = np.reshape(xhat.T, (N, C, H, W))
    #print("gamma[...].shape: ",gamma[np.newaxis, :, np.newaxis, np.newaxis].shape )  # (1, 1, 1, 1, 6, 1, 1)
    out = gamma * xhat + beta # nxb edited this line to conform to the latest update from the cs231n TAs/Ph.D students

    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps, G)










    """    { // begin my code   (doesn't work, misunderstands the required dimensions, etc.)
    (N, C, H, W) = x.shape
    assert C%G ==0,   "  check your G and x values, dumbass.  If you need a tutorial on how to 'get good,'  there's one [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ)"
    GS=groupsize   = int(C / G)
    outs=[]
    caches=[]
    bn_param=deepcopy(gn_param)
    bn_param['eps']=eps
    xs= np.array_split(x, G, axis=1)  # "xs"   is a list of np.arrs.
    for g in range(G):  
      x = xs[g]   # xs[g].shape == (N, C / G,  H,W)
      #print("before passing to 'batchnorm_forward()', \n x.shape: ", x.shape)
      x = x.transpose((1,2,3,0))
      #print("before passing to 'batchnorm_forward()', \n x.shape: ", x.shape)
      x = x.reshape((GS*H*W, N))
      print("before passing to 'batchnorm_forward()', \n x.shape = ", x.shape)
      print(" gamma.shape == {0}, and \n beta.shape == {1}".format(gamma.shape,  beta.shape))

      out_partial, cache = batchnorm_forward(x, gamma, beta, bn_param)

      #===================================================================================================
      #                     within function "spatial_groupnorm_fwd()" :
      #===================================================================================================
      print("AFTER passing to 'batchnorm_forward()', \n out_partial.shape: ", out_partial.shape)
      # the ordering is right   when I use "lis.append(x)" (it's the same as the ordering of "xs")  :
      out_partial = out_partial.reshape((GS,H,W,N))
      out_partial = out_partial.transpose((3,0,1,2))
      outs.append(out_partial)
      caches.append(cache)

      # reference from spatial_batchnorm_forward():   
      #out, cache = batchnorm_forward(x.transpose((0,2,3,1)).reshape((N*H*W, C)), gamma, beta, bn_param)   # out.shape == (40,3)  
      #out = out.reshape((N,H,W,C)).transpose((0,3,1,2)) 

      # TODO:   finish this.   at the end, I gotta combine  all the gradients back together with   np.concatenate()   g         

    out = np.concatenate((tuple(outs)), axis=1)
    print("out.shape: ", out.shape)
    """     # }  // end my code











    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache
#================== end func definition of  spatial_groupnorm_forward(x, gamma, beta, G, gn_param):==================


#==============================================================================
def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    # Version of the code  copied from Haofei Xu:  ( URL: https://github.com/haofeixu/stanford-cs231n-2018/blob/master/assignment2/cs231n/layers.py )
    N, C, H, W = dout.shape

    xhat, gamma, xmu, ivar, sqrtvar, var, eps, G = cache

    dxhat = dout * gamma[np.newaxis, :, np.newaxis, np.newaxis]

    # Set keepdims=True to make dbeta and dgamma's shape be (1, C, 1, 1)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout*xhat, axis=(0, 2, 3), keepdims=True)

    # Reshape and transpose back
    dxhat = np.reshape(dxhat, (N*G, C//G*H*W)).T
    xhat = np.reshape(xhat, (N*G, C//G*H*W)).T
    Nprime, Dprime = dxhat.shape
    dx = 1.0/Nprime * ivar * (Nprime*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))
    dx = np.reshape(dx.T, (N, C, H, W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
#================== end func definition of   spatial_groupnorm_backward(dout, cache):==================


#==============================================================================
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx
#================== end func definition of   svm_loss(x, y):==================


#==============================================================================
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    # there's some mathemagic going on in the above line of code ("log_probs = shifted_logits - np.log(Z)").  how does -np.log(Z)   give you the logprobs when we didn't already take the log() to get shifted_logits ??     -nxb

    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
#================== end func definition of   softmax_loss(x, y):==================



































































































