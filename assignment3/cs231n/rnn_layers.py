from __future__ import print_function, division
from builtins import range
import numpy as np

# nxb's imports:
from collections import OrderedDict
from debug import debug


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


#===================================================================================================
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #===================================================================================================
    #  Don't we want to get rid of the "N" dimension somehow?   -nxb, Aug 2, 2019
    #    The network shouldn't have to know about the minibatch size 'N'.
    #===================================================================================================

    next_h  = np.tanh(prev_h.dot(Wh)      + x.dot(Wx)  + b)
    cache   = (x, prev_h, Wx, Wh, b, next_h)                                                    #      __
    # We should really use a dictionary instead of a tuple here.  But they said to use a tuple...  ¯\_(ツ)_/¯    -nxb
                                                                                                #      TT

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache
#===================================================================================================
#===================================================================================================











#===================================================================================================
def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack & prep for calculation:
    (x, prev_h, Wx, Wh, b, next_h)  = cache
    # dtanh= 1 - (tanh ** 2)
    dmid  = (1 - (next_h**2) ) * dnext_h
    DOWN= 0

    db  = np.sum(dmid, axis=DOWN) # db.shape is correct here.  --nxb, Aug 2, 2019
    # dmid.dot(...) :
    dx  = dmid.dot(Wx.T)
    dprev_h = dmid.dot(Wh.T)
    # (...) .dot(dmid)
    dWx = (x.T).dot(dmid)
    dWh =  (prev_h.T).dot(dmid)  #(dmid.T).dot(prev_h)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db
#=========== end func definition of "rnn_step_backward(dnext_h, cache):" ============
#===================================================================================================

#===================================================================================================
def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # NOTE:  "h[:,-1,:]=h0" is a little hack so I don't have to write funky code in the loop.  Not the clearest until you realize why I did it.  Basically, we want each timestep to be defined in terms of the hidden layers' values at another timestep, but I need to put in the initial value somehow to define h @ t=0.  Then at time t==T-1, h[:,t,:]=... overwrites the initial value of h[:,-1,:].  -nxb
    # TODO:  understand this properly.



    N,T,D,H = (*x.shape, h0.shape[1])        # I like this way of unpacking dimensions much better.  TODO: put this style/syntax everywhere -nxb
    cache=OrderedDict()
    h=np.zeros((N,T,H),dtype='float64')
    h[:,0,:], cache[0]= rnn_step_forward(x[:,0,:], h0, Wx, Wh, b) # get next value from current time step's val
    for t in range(1,T):
      h[:,t,:], cache[t]= rnn_step_forward(x[:,t,:], h[:,t-1,:], Wx, Wh, b)  # get next value from current time step's val
    #=============================================================================

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache
#===================================================================================================
#===================================================================================================


#===================================================================================================
def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    - cache: from the function "rnn_forward()"
 
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,T,H = dh.shape
    X     = 0
    D     = cache[0][X].shape[-1]

    # Initialize all partial derivatives:
    dx    = np.zeros((N,T,D), dtype="float64")
    dh0   = np.zeros((N,  H), dtype="float64")
    dWx   = np.zeros((D,  H), dtype="float64")
    dWh   = np.zeros((H,  H), dtype="float64")
    db    = np.zeros((H,   ), dtype="float64")

    # "prev" is technically "after" in the time-series-nature of the computational graph.  see page 31 of the 2018 slides [here](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture10.pdf) for a visualization of this;  the upstream gradients from h_greater are chronologically "after" the current hidden unit & all its inputs & outputs.
    dprev_h = np.zeros((N,H),dtype='float64')
    #==================================================================
    # Backpropagate BACKWARDS through time:
    #   (ie. from most recent time step to oldest time step)
    #==================================================================
    for t in range(T)[::-1]:  # here, "[::-1]"   means "backward"
      # See the "NOTE" for how to do the dh calculation:
      dx[:,t,:], dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dprev_h + dh[:,t,:], cache[t])
      dWx +=  dWx_t
      dWh +=  dWh_t
      db  +=  db_t
    dh0=dprev_h

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db
#==================== end function def of "rnn_backward(dh, cache):"====================
#===================================================================================================











#===================================================================================================
def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out=W[x,:]
    cache=(x,W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache
#====================== end function  def of  "word_embedding_forward(x, W):"======================
#===================================================================================================
  





#===================================================================================================
def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,W=cache                                   #dx = dout.sum(axis=2)
    dW=np.zeros(W.shape, dtype=W.dtype)
    np.add.at(dW, x, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW
#=================== end function  def of  "word_embedding_backward(dout, cache):"===================
#===================================================================================================








#===================================================================================================
def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)
#===================================================================================================
#================================== end func def of " sigmoid(x):"==================================
#===================================================================================================


#===================================================================================================
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack & label dimensions:
    N,D,H = (*x.shape, prev_h.shape[1])

    # Activation vector:
    #   (N, 4H)     (N, 4H)       (4H,)
    a = x.dot(Wx) + prev_h.dot(Wh) + b

    # Divide "a" into 4 vectors ("ifog")
    # (N, H)
    ai, af, ao, ag = np.array_split(a, 4, axis=1)
    assert ai.shape == af.shape == ao.shape == ag.shape #== (N, H)
    # Nonlinearity:
    i, f, o, g = sigmoid(ai), sigmoid(af), sigmoid(ao), np.tanh(ag)
    input_gate, forget_gate, output_gate, block_input = i,f,o,g
    next_c = (f*prev_c)  + (i*g)
    next_h = o * np.tanh(next_c)
    cache = (x, prev_h, prev_c, next_h, next_c, Wx, Wh,
     b, a, input_gate, forget_gate, output_gate, block_input)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache
#===================================================================================================
#================== end func definition of  "lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):"==================
#===================================================================================================















































#===================================================================================================
def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, H = dnext_h.shape

    # Unpack cache variables:
    (x, prev_h, prev_c, next_h, next_c, Wx, Wh,
     b, a, input_gate, forget_gate, output_gate, block_input) = cache

    dnext_c += dnext_h*output_gate*(1-np.tanh(next_c)**2)
    dprev_c = dnext_c*forget_gate
    dinput_gate = dnext_c*block_input
    dblock_input = dnext_c*input_gate
    dforget_gate = dnext_c*prev_c
    doutput_gate = dnext_h*np.tanh(next_c)

    # Backpropagate through nonlinearities: sigmoid() for most  and tanh() for "g" gate :
    # Initialize matrix for the affine transformation:
    da = np.zeros((N, 4*H))
    da[:,0:H] = dinput_gate*input_gate*(1-input_gate)
    da[:,H:2*H] = dforget_gate*forget_gate*(1-forget_gate)
    da[:,2*H:3*H] = doutput_gate*output_gate*(1-output_gate)
    da[:,3*H:] = dblock_input*(1-block_input**2)

    # Vanilla affine partial derivatives that we've seen a million times before   (places we've seen them: fc_fwd(), out=x.dot(W)+b )
    dx = np.dot(da,Wx.T)
    dWx = np.dot(x.T, da)
    dWh = np.dot(prev_h.T, da)
    dprev_h = np.dot(da, Wh.T)
    db = np.sum(da, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db
#===================================================================================================

#===================================================================================================
def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Initialization:
    N,T,D,H = (*x.shape, h0.shape[1])        # I like this way of unpacking dimensions better.  TODO: put this style/syntax everywhere -nxb
    cache=OrderedDict()
    h=np.zeros((N,T+1,H),dtype='float64')

    # Initialize 0th hidden and cell state:
    h[:,0,:]=h0
    prev_c=np.zeros((h0.shape), dtype=h0.dtype)

    # Forward pass:
    for t in range(T):
      h[:,t+1,:], prev_c, cache[t]= lstm_step_forward(x[:,t,:], h[:,t,:], prev_c, Wx, Wh, b)  # get next value from current time step's val

    # Remove state "h0"
    h=h[:,1:,:]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


#=====================================================================================================
def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass  ( the function "lstm_forward(...)" )

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get dimensions:
    X     = 0 
    N,T,H, D = (*dh.shape, cache[0][X].shape[-1])

    # Initialize all partial derivatives:
    dx    = np.zeros((N,T,D), dtype="float64")
    dh0   = np.zeros((N,  H), dtype="float64")
    dWx   = np.zeros((D,4*H), dtype="float64")
    dWh   = np.zeros((H,4*H), dtype="float64")
    db    = np.zeros((4*H, ), dtype="float64")

    # Some documentation from nxb, Fri Aug  9 19:12:48 EDT 2019:
    # "prev" is technically "after" in the time-series-nature of the computational graph.  see page 31 of the 2018 slides [here](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture10.pdf) for a visualization of this;  the upstream gradients from h_greater are chronologically "after" the current hidden unit & all its inputs & outputs.
    # We initialize dc (AKA "dprev_c") to 0 b/c the last cell value has NO IMPACT on the value of the final loss function.  There are very much **internal** values of "dc" or "dprev_c," but those values of "dprev_c" are all due to the loss' interaction with the hidden state "h."  -nxb, Fri Aug  9 19:04:13 EDT 2019
    dprev_c = np.zeros((N,H),dtype="float64")
    dprev_h = np.zeros((N,H),dtype="float64")

    #==================================================================
    # Backpropagate BACKWARDS through time:
    #   (ie. from most recent time step to oldest time step)
    #==================================================================
    for t in range(T)[::-1]:  # here, "[::-1]"   means "backward"
      # See the "NOTE" for how to do the dh calculation:
      dx[:,t,:], dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(dprev_h + dh[:,t,:], dprev_c, cache[t])
      dWx +=  dWx_t
      dWh +=  dWh_t
      db  +=  db_t
    dh0 = dprev_h

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db
#=====================================================================================================
#====================================def lstm_backward(dh, cache):====================================
#=====================================================================================================


#===================================================================================================
def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache
#===================================================================================================


#===================================================================================================
def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db
#===================================================================================================


#===================================================================================================
def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    # The "- np.max(x_flat)" in this line   is just a technique to control the explosion of values when we go "np.exp()," right?
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    # e ** ( x - max(x))   (The above line basically means "e to the x"  -nxb, Aug 4, 2019)   "probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))"  was the above line if the code changes
    probs /= np.sum(probs, axis=1, keepdims=True)
    # Divide by sums of exps (http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture03.pdf)
    loss = -np.sum(  mask_flat *    np.log( probs[np.arange(N * T), y_flat] )  ) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
#===================================================================================================
#================== end func definition of " temporal_softmax_loss(x, y, mask, verbose=False):"==================
#===================================================================================================

































































































