"""Useful functions for processing images, particularly for Tensorboard."""

import numpy as np


def normalize_0_1(x):
    max_x = np.max(x)
    min_x = np.min(x)
    return (x.astype(np.float32) - min_x) / (max_x - min_x)


def _pad_images(images, padding_value=0.5, padding_width=2):
    """Pad images to create frame around them.

    Args:
        images: Numpy array of shape [batch_size, height, width] or
            [batch_size, channels, height, width].
        padding_value: Float. Value of the padding.
        padding_width: Int. Width of the padding.

    Returns:
        padded_images: Numpy array containing padded images.
    """
    image_dim = len(images.shape)
    paddings_1d = (padding_width, padding_width)
    if image_dim == 3:
        paddings = [(0, 0), paddings_1d, paddings_1d]
    elif image_dim == 4:
        paddings = [(0, 0), (0, 0), paddings_1d, paddings_1d]
    else:
        raise ValueError(
            'Image dimension is {} but must be 3 or 4.'.format(image_dim))
    padded_images = np.pad(images, pad_width=paddings,
                           constant_values=padding_value)
    return padded_images


def images_to_grid(images,
                   grid_height=4,
                   grid_width=4,
                   padding_value=0.5):
    """Tile images into a grid with padding between them.

    Essentially this function takes the first grid_height * grid_width elements
    from the images batch and reshapes them.

    Args:
        images: Numpy array of shape [batch_size, height, width] or
            [batch_size, channels, height, width].
        grid_height: Int. Height of the resulting image grid.
        grid_height: Int. Width of the resulting image grid.
        padding_value: Float. Value of the padding between the images.

    Returns:
        images: Numpy array containing images tiled into a grid with padding
            between.
    """

    # Append blank frames if batch_size < grid_height * grid_width
    if images.shape[0] < grid_height * grid_width:
        num_blank_frames = grid_height * grid_width - images.shape[0]
        blank_frames_shape = (num_blank_frames,) + images.shape[1:]
        blank_frames = padding_value * np.ones(blank_frames_shape,
                                               dtype=images.dtype)
        images = np.concatenate([images, blank_frames], axis=0)

    images = images[:grid_height * grid_width]
    images = _pad_images(images, padding_value=padding_value)
    images_shape = list(images.shape)
    height, width = images_shape[-2:]
    new_images_shape = ([grid_height, grid_width] +
                        (4 - len(images_shape)) * [1] + images_shape[1:])
    images = np.reshape(images, new_images_shape)
    images = np.transpose(images, axes=[2, 0, 3, 1, 4])
    images = np.reshape(
        images, [-1, grid_height * height, grid_width * width])
    return images
