from core import *

# noinspection PyProtectedMember
_FLOATX = Config._FLOATX
# noinspection PyProtectedMember
_DATATYPES = Config._DATATYPES


def binary_cross_entropy(prediction, target, weights=None, with_logits=True, aggregate=True,
                         aggregation_mode='mean'):
    """
    Computes the binary cross entropy given `prediction` and `target` tensors,
    where `prediction` must be the output of a linear layer when `with_logits`
    is set to true (default). If `with_logits` is set to False, prediction is assumed
    to be between 0 and 1 (i.e. the output of a sigmoid layer).

    If `prediction` and `target` are 4 or 5 dimensional image tensors, they're
    converted to matrices with `Antipasti.backend.image_tensor_to_matrix` function.

    :type prediction: tensorflow.Tensor or tensorflow.Variable
    :param prediction: Prediction tensor. Values in it must remain between 0 and 1
                       if `with_logits` is set to `False`.

    :type target: tensorflow.Tensor or tensorflow.Variable
    :param target: Target tensor.

    :type weights: tensorflow.Tensor or tensorflow.Variable
    :param weights: Pixel-wise weight tensor. Should have the same shape as `prediction`
                    or `target`.

    :type with_logits: bool
    :param with_logits: Whether `prediction` is a tensor of logits.

    :type aggregate: bool
    :param aggregate: Whether to aggregate the loss (which is initially a vector of shape
                      (batchsize,)) to a scalar.


    :type aggregation_mode: str
    :param aggregation_mode: If `aggregate`, the aggregation (reduction) mode to be used.

    :return: Binary cross entropy vector or scalar
    """
    # Flatten to matrix
    prediction_flattened = image_tensor_to_matrix(prediction)
    target_flattened = image_tensor_to_matrix(target)
    weights_flattened = image_tensor_to_matrix(weights) if weights is not None else None
    # Compute loss
    if with_logits:
        # Remember that binary cross entropy is elementwise (unlike softmax cross entropy)
        bce_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_flattened,
                                                             targets=target_flattened, name='bce')
        # Weight bce_matrix. Since tf.mul supports broadcasting, `weights` may or may not be
        # multi-channel. ;-)
        if weights_flattened is not None:
            bce_matrix = multiply(weights_flattened, bce_matrix, name='bce_weighting')
        # Sum along the channel axis
        bce_vector = reduce_(bce_matrix, 'sum', axis=1)
        # Aggregate if required
        if aggregate:
            bce = reduce_(bce_vector, aggregation_mode, axis=0)
        else:
            bce = bce_vector
    else:
        raise NotImplementedError("Binary cross entropy without logits is yet to be implemented.")
    # Done.
    return bce


def sorensen_dice_distance(prediction, target, weights=None, with_logits=True, aggregate=True,
                           aggregation_mode='sum'):
    """
    Computes the softened Sorensen-Dice distance given a `prediction` and a `target`.
    It is $(1 - SDC)$, where $SDC$ is the Sorensen-Dice coefficient.

    It's usually assumed that `prediction` and `target` are binary, but this function does
    not require them to be. If `with_logits` is set to true, an elementwise sigmoid will first be
    applied to the predictions. The keyword `aggregate` has to be true by definition of the
    coefficient.

    :type prediction: tensorflow.Tensor or tensorflow.Variable
    :param prediction: Prediction tensor.
                       Values must be between 0 and 1 if `with_logits` is set to False.

    :type target: tensorflow.Tensor or tensorflow.Variable
    :param target: Target tensor.

    :type weights: tensorflow.Tensor or tensorflow.Variable
    :param weights: Pixel-wise weight tensor. Should have the same shape as `prediction`
                    or `target`.

    :type with_logits: bool
    :param with_logits: Whether `prediction` is a tensor of logits, in which case it will be
                        passed through a sigmoid first.

    :type aggregate: bool
    :param aggregate: Whether to aggregate to a scalar. This has to be true.

    :type aggregation_mode: str
    :param aggregation_mode: If `aggregate`, the aggregation (reduction) mode to use.
                             This is defined by the coefficient itself and can not be changed.

    :return: Sorensen dice distance.
    """
    assert aggregation_mode == 'sum', \
        "Aggregation mode is predefined for the Sorensen-Dice coefficient (= 'sum'). " \
        "Got {} instead.".format(aggregation_mode)
    assert aggregate, \
        "Sorensen-Dice coefficient aggregates by definition, " \
        "cannot have the aggregate keyword set to `False`."

    with ContextSupermanager(name_scope='sorensen_dice_distance').manage():
        if with_logits:
            # Pass through a sigmoid
            prediction = sigmoid(prediction)

        if weights is not None:
            # Weight prediction and targets
            prediction = multiply(weights, prediction, name='prediction_weighting')
            target = multiply(weights, target, name='target_weighting')

        # Compute dice coefficient with safe divide
        sorensen_dice_coefficient = 2 * divide(reduce_(multiply(prediction, target), mode='sum'),
                                               reduce_(pow(prediction, 2), mode='sum') +
                                               reduce_(pow(target, 2), mode='sum'),
                                               safe=True)

        # Compute distance as 1 - coeff
        distance = 1 - sorensen_dice_coefficient
        # Done.
        return distance


def tversky_distance(prediction, target, weights=None, alpha=1., beta=1., with_logits=True,
                     aggregate=True, aggregation_mode='sum'):
    """
    Computes the softened Tversky distance given a `prediction` and a `target`.
    It is $(1 - TI)$, where TI$ is the Tversky index.

    It's usually assumed that `prediction` and `target` are binary, but this function does
    not require them to be. If `with_logits` is set to true, an elementwise sigmoid will first be
    applied to the predictions. The keyword `aggregate` has to be true by definition of the
    Tversky index.

    :type prediction: tensorflow.Tensor or tensorflow.Variable
    :param prediction: Prediction tensor.
                       Values must be between 0 and 1 if `with_logits` is set to False.

    :type target: tensorflow.Tensor or tensorflow.Variable
    :param target: Target tensor.

    :type weights: tensorflow.Tensor or tensorflow.Variable
    :param weights: Pixel-wise weight tensor. Should have the same shape as `prediction`
                    or `target`.

    :type with_logits: bool
    :param with_logits: Whether `prediction` is a tensor of logits, in which case it will be
                        passed through a sigmoid first.

    :type aggregate: bool
    :param aggregate: Whether to aggregate to a scalar. This has to be true.

    :type aggregation_mode: str
    :param aggregation_mode: If `aggregate`, the aggregation (reduction) mode to use.
                             This is defined by the coefficient itself and can not be changed.

    :return: Tversky distance.
    """
    assert aggregation_mode == 'sum', \
        "Aggregation mode is predefined for the Tversky distance (= 'sum'). " \
        "Got {} instead.".format(aggregation_mode)
    assert aggregate, \
        "Tversky distance aggregates by definition, " \
        "cannot have the aggregate keyword set to `False`."

    with ContextSupermanager(name_scope='tversky_distance').manage():
        if with_logits:
            # Pass through a sigmoid
            prediction = sigmoid(prediction)

        if weights is not None:
            # Weight prediction and targets
            prediction = multiply(weights, prediction, name='prediction_weighting')
            target = multiply(weights, target, name='target_weighting')

        # Put together tversky index
        y_times_yt = reduce_(multiply(prediction, target), mode='sum')
        y_times_one_minus_yt = reduce_(multiply(prediction, 1. - target), mode='sum')
        one_minus_y_times_yt = reduce_(multiply(1. - prediction, target), mode='sum')

        tversky_index_numerator = y_times_yt
        tversky_index_denominator = y_times_yt + \
                                    alpha * y_times_one_minus_yt + \
                                    beta * one_minus_y_times_yt
        tversky_index = divide(tversky_index_numerator, tversky_index_denominator, safe=True)

        # Compute distance as 1 - tversky index
        distance = 1. - tversky_index
        # Done
        return distance


def binary_accuracy(prediction, target, prediction_threshold=0.5, target_threshold=0.5):
    """
    Computes the binary accuracy of the `prediction` with respect to the given `target`.
    The `prediction` and `target` are thresholded at `prediction_threshold` and `target_threshold`
    respectively.

    :type prediction: tensorflow.Tensor or tensorflow.Variable
    :param prediction: Prediction tensor.

    :type target: tensorflow.Tensor or tensorflow.Variable
    :param target: Target tensor.

    :type prediction_threshold: float or tensorflow.Tensor or tensorflow.Variable
    :param prediction_threshold: Threshold for prediction.

    :type target_threshold: float or tensorflow.Tensor or tensorflow.Variable
    :param target_threshold: Threshold for target.

    :return: Binary accuracy scalar.
    """
    # Threshold prediction and target
    thresholded_prediction = threshold_tensor(prediction, prediction_threshold,
                                              name='prediction_threshold')
    thresholded_target = threshold_tensor(target, target_threshold,
                                          name='target_threshold')
    # Compare prediction and target and reduce_mean
    _binary_accuracy = reduce_(equal(thresholded_prediction, thresholded_target, as_dtype=_FLOATX,
                                     name='bin_accuracy_comparison'),
                               mode='mean', name='bin_accuracy_reduction')
    # Return binary accuracy scalar
    return _binary_accuracy


def frequency_distribution(tensor, normalize=True, min_tensor_value=0., max_tensor_value=1.,
                           n_bins=10, dtype=_FLOATX, epsilon=1e-8):
    """
    Computes the frequency distribution of `tensor`.

    :type tensor: tensorflow.Tensor or tensorflow.Variable
    :param tensor: A tensor.

    :type normalize: bool
    :param normalize: Whether to normalize the frequency distribution.
                      A constant `epsilon` is added to the unnormalized frequency
                      distribution to avoid having zeros in the output distribution
                      vector to make it numerically stable for e.g. entropy computations.

    :type min_tensor_value: float or tensorflow.Tensor or tensorflow.Variable
    :param min_tensor_value: The smallest possible value in `tensor`.

    :type max_tensor_value: float or tensorflow.Tensor or tensorflow.Variable
    :param max_tensor_value: The largest possible value in `tensor`.

    :type n_bins: int
    :param n_bins: Number of bins to compute the histogram with.

    :type dtype: str
    :param dtype: Output datatype.

    :type epsilon: float
    :param epsilon: A small float value for numerical stability.

    :return: Frequency distribution as a floatX vector.
    """
    # Parse dtype
    dtype = to_tf_dtype(dtype)
    # Compute histogram
    hist = tf.histogram_fixed_width(tensor,
                                    value_range=[min_tensor_value, max_tensor_value],
                                    nbins=n_bins, dtype=dtype)
    # Normalize histogram if required
    if normalize:
        hist = divide(hist + epsilon, reduce_((hist + epsilon), 'sum'))
    # Done
    return hist


def shannon_entropy(tensor, min_tensor_value=0., max_tensor_value=1., n_bins=10):
    """
    Computes the shannon entropy of the frequency distribution of `tensor`.

    :type tensor: tensorflow.Tensor or tensorflow.Variable
    :param tensor: A tensor.

    :type min_tensor_value: float or tensorflow.Tensor or tensorflow.Variable
    :param min_tensor_value: The smallest possible value in `tensor`.

    :type max_tensor_value: float or tensorflow.Tensor or tensorflow.Variable
    :param max_tensor_value: The largest possible value in `tensor`.

    :type n_bins: int
    :param n_bins: Number of bins to compute the histogram with.

    :return: Shannon entropy scalar.
    """
    # Compute normalized histogram
    normalized_hist = frequency_distribution(tensor, normalize=True,
                                             min_tensor_value=min_tensor_value,
                                             max_tensor_value=max_tensor_value, n_bins=n_bins)
    # Compute shannon entropy
    _shannon_entropy = reduce_(-log(normalized_hist) * normalized_hist, 'sum')
    # Done
    return _shannon_entropy


def kullback_leibler_divergence_of_frequency_distributions(tensor_p, tensor_q):
    """
    Computes KL-Divergence of the frequency distributions of `tensor_p` and `tensor_q`.
    Assumes both `tensor_p` and `tensor_q` have values between 0. and 1.
    """
    # Compute normalized frequency distributions (NFD's)
    p = frequency_distribution(tensor_p, normalize=True)
    q = frequency_distribution(tensor_q, normalize=True)
    # Compute KL-Divergence
    kl_divergence = reduce_(p * log(divide(p, q)), 'sum')
    return kl_divergence


def jensen_shannon_divergence_of_frequency_distributions(tensor_p, tensor_q):
    """
    Computes the Jensen-Shannon divergence of the frequency distributions of
    tensor_p and tensor_q.
    """
    # Compute normalized frequency distributions (NFD's)
    p = frequency_distribution(tensor_p, normalize=True)
    q = frequency_distribution(tensor_q, normalize=True)
    m = 0.5 * (p + q)
    # Compute KL-Divergences
    kl_divergence_pm = reduce_(p * log(divide(p, m)), 'sum')
    kl_divergence_qm = reduce_(q * log(divide(q, m)), 'sum')
    # Compute JS-Divergence
    js_divergence = 0.5 * (kl_divergence_pm + kl_divergence_qm)
    return js_divergence
