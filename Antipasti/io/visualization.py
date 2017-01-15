__author__ = "Nasim Rahaman"

from scipy.misc import toimage
import os


def tensor_dump(tensor, dump_directory, batches=None, channels=None, name_prefix=None,
                scale_between_zero_and_one=True, make_folder=True):
    """Dumps a given (numpy) tensor (as BHWC or BDHWC) to file as images."""
    # TODO Full documentation
    # Make a folder if required
    if make_folder:
        folder_name = name_prefix if name_prefix is not None else "tensor"
        final_dump_directory = os.path.join(dump_directory, folder_name)
        os.mkdir(final_dump_directory, 0755)
    else:
        final_dump_directory = dump_directory

    # Parse batches and channels
    batches = range(tensor.shape[0]) if batches is None else batches
    channels = range(tensor.shape[-1]) if channels is None else channels

    # If tensor is 5D, the 2D slices of the volume are to be printed individually
    if tensor.ndim == 5:
        Ts = range(tensor.shape[1])
    else:
        Ts = []

    # Parse cmin and cmax
    if scale_between_zero_and_one:
        cmin = 0.0
        cmax = 1.0
    else:
        cmin = None
        cmax = None

    # Parse name_prefix
    name_prefix = '' if name_prefix is None else name_prefix

    # Loop and save
    for batch in batches:
        for channel in channels:
            if tensor.ndim == 5:
                for T in Ts:
                    file_name = os.path.join(final_dump_directory,
                                             "{}--batch-{}--channel-{}--T-{}.png".
                                             format(name_prefix, batch, channel, T))
                    toimage(tensor[batch, T, ..., channel], cmin=cmin, cmax=cmax).save(file_name)
            else:
                file_name = os.path.join(final_dump_directory,
                                         "{}--batch-{}--channel-{}.png".
                                         format(name_prefix, batch, channel))
                toimage(tensor[batch, ..., channel], cmin=cmin, cmax=cmax).save(file_name)
