import numpy as np


def perplexity_from_token_probs(token_probs):
    '''Calculate perplexity from token probabilities
    '''
    if isinstance(token_probs, list):
        token_probs = np.array(token_probs)
    assert np.min(token_probs) >= 0, 'Token probabilities must be non-negative, the smallest is {}'.format(np.min(token_probs))
    assert np.max(
        token_probs) <= 1, 'Token probabilities must be less than or equal to 1'
    return 2 ** (-np.mean(np.log2(token_probs)))
