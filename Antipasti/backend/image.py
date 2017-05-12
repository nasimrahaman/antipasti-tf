from core import *
import tensorflow as tf

# noinspection PyProtectedMember
_FLOATX = Config._FLOATX
# noinspection PyProtectedMember
_DATATYPES = Config._DATATYPES


def image_tensor_to_matrix(tensor):
    """
    Convert an image tensor (as BHWC or BDHWC) to a matrix of shape (B * H * W, C).
    Adds the known original shape as a field in antipasti collection.
    Note that this function works as expected (though not without added redundancy) even when
    `tensor` is a matrix already.
    """
    # Log original shape
    shape_before_flattening = shape(tensor, symbolic=False)
    # Get symbolic value for the number of channels
    num_channels = shape(tensor, symbolic=True)[-1]
    # Flatten
    flat_matrix = reshape(tensor, shape=[-1, num_channels], name='flatten_image_tensor_to_matrix')
    # Have original shape as a field in antipasti collection
    # (such that the flat matrix can in principle be unflattened)
    py2.add_to_antipasti_collection(flat_matrix, shape_before_flattening=shape_before_flattening)
    # Done.
    return flat_matrix
