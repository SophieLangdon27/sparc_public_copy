# Python code to run Sparse Regression Code (SPARCs) simulations
#
# Copyright (c) 2020 Kuan Hsieh

import numpy as np
from sparc_sophie import sparc_encode_ldpc, sparc_decode_posterior_probs

def sparc_posterior_probs(code_params, decode_params, awgn_var, ldpc_vec, rand_seed=None):

    """
    End-to-end simulation of Sparse Regression Code (SPARC) encoding/decoding
    in AWGN channel.
    """

    # Currently cheating as encoder directly passes fast transforms Ab and Az to
    # the deocder. (Decoder doesn't use random seed to generate fast transform.)

    # Simulation
    bits_i, beta0, x, Ab, Az = sparc_encode_ldpc(code_params, awgn_var, rand_seed, ldpc_vec)
    y                        = awgn_channel(x, awgn_var, rand_seed)
    bits_o, beta, T, nmse, expect = sparc_decode_posterior_probs(y, code_params, decode_params,
                                                 awgn_var, rand_seed, beta0, Ab, Az)

    return beta


######## Channel models ########

def awgn_channel(input_array, awgn_var, rand_seed):
    '''
    Adds Gaussian noise to input array

    Real input_array:
        Add Gaussian noise of mean 0 variance awgn_var.

    Complex input_array:
        Add complex Gaussian noise. Indenpendent Gaussian noise of mean 0
        variance awgn_var/2 to each dimension.
    '''

    assert input_array.ndim == 1, 'input array must be one-dimensional'
    assert awgn_var >= 0

    rng = np.random.RandomState(rand_seed)
    n   = input_array.size

    if input_array.dtype == np.float:
        return input_array + np.sqrt(awgn_var)*rng.randn(n)

    elif input_array.dtype == np.complex:
        return input_array + np.sqrt(awgn_var/2)*(rng.randn(n)+1j* rng.randn(n))

    else:
        raise Exception("Unknown input type '{}'".format(input_array.dtype))

