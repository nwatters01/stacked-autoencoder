"""Classes for convolutional networks.

This file contains a few classes and functions for 2D convolutional networks.
There are two important pieces of functionality that these classes implement on
top of pytorch's convolutional classes:
1. Recording the spatial shape of their inputs and outputs. Recording these is
    useful for things like padding and deconvolution.
2. Same-padding, namely for these conv and devonv classes the user can ask
    them to automatically pad such that the output has the same spatial shape as
    the input divided by the stride (so the class takes into account the kernel
    size for this computation).
"""

import abc
import torch
import six


@six.add_metaclass(abc.ABCMeta)
class Abstract2D(torch.nn.Module):
    """Abstract2D model.

    All 2D convolutional classes should inherit from this.
    """

    def __init__(self):
        """Create Abstract2D module."""
        super(Abstract2D, self).__init__()

        # Note: It is important to initialize input_spatial_shape and
        # output_spatial_shape as dictionaries, not as None, because these
        # dictionaries can be passed into functions/constructors and their
        # values will be propagated later. This lazy computation is necessary
        # for automatic padding in deconvolutions, for example.
        self._input_spatial_shape = {
            'height': None,
            'width': None,
        }
        self._output_spatial_shape = {
            'height': None,
            'width': None,
        }
        self._space_evaluated = False

    @abc.abstractmethod
    def _forward(self, x):
        """Forward application of the module.

        This must be implemented by the subclass.

        Args:
            x: Tensor with shape [batch, channels, height, width].
        """

    def forward(self, x):
        """Wrapper for the forward application of the module.

        This calls self._forward(x) and records the input/output spatial shapes.

        Args:
            x: Tensor with shape [batch, channels, height, width].
        """

        out_x = self._forward(x)
        if not self.space_evaluated():
            self._input_spatial_shape['height'] = x.shape[2]
            self._input_spatial_shape['width'] = x.shape[3]
            self._output_spatial_shape['height'] = out_x.shape[2]
            self._output_spatial_shape['width'] = out_x.shape[3]
            self._space_evaluated = True

        return out_x

    @property
    def input_spatial_shape(self):
        """Dictionary with keys ['height', 'width']."""
        return self._input_spatial_shape

    @property
    def output_spatial_shape(self):
        """Dictionary with keys ['height', 'width']."""
        return self._output_spatial_shape

    def space_evaluated(self):
        """Returns a Boolean, whether spatial shapes have been evaluated."""
        return self._space_evaluated


class Conv2D(Abstract2D):
    """Convolutional layer analogous torch.nn.Conv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='SAME'):
        """Construct convolutional layer.

        Args:
            in_channels: Int. Number of input channels.
            out_channels: Int. Number of output channels.
            kernel_size: Int or iterable of length 2. If int, represents the
                kernel size in each of the spatial dimensions.
            stride: Int or iterable of length 2. If int, represents the stride
                in each of the spatial dimensions.
            padding: Iterable of non-negative ints of length 4 or 'SAME' or
                'VALID'.
                * If iterable, elements represent the zero-padding to the left,
                    right, top, and bottom of the input respectively.
                * If 'SAME', padding is automatically constructed to ensure that
                    the output spatial shape is the input spatial shape divided
                    by the stride.
                * If 'VALID', no padding is used.
        """
        super(Conv2D, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, (list, tuple)):
            padding = padding
        elif padding == 'VALID':
            padding = (0, 0, 0, 0)
        elif padding == 'SAME':
            top_padding = kernel_size[0] // 2
            bottom_padding = kernel_size[0] - top_padding - 1
            left_padding = kernel_size[1] // 2
            right_padding = kernel_size[1] - left_padding - 1
            padding = (
                left_padding, right_padding, top_padding, bottom_padding
            )
        else:
            raise ValueError('Invalid padding {}.'.format(padding))
        self._padding_module = torch.nn.ZeroPad2d(padding)

        self._layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        # Remaining unused input in each dimension from striding. Will be
        # propagated at the first .forward() call.
        self._remainder = {
            'height': None,
            'width': None,
        }

    def _forward(self, x):
        """Forward application of the module.

        Args:
            x: Tensor with shape [batch, channels, height, width].

        Returns:
            out_x: Output tensor with shape
                [batch, self._out_channels, output_height, output_width].
        """
        padded_x = self._padding_module.forward(x)

        if not self.space_evaluated():
            # Evaluate remainder. Remainder is the spatial border that is
            # ignored by the convolution. If stride is 1 then the remainder must
            # be zero. The remainder may also be zero when stride is not 1,
            # depending on the stride, padding, and input spatial shape.
            # Recording this remainder is useful when constructing a
            # deconvolutional layer symmetric to this convolutional layer.
            height = padded_x.shape[2]
            width = padded_x.shape[3]
            remainder_height = (
                (height - self._kernel_size[0]) % self._stride[0])
            remainder_width = (
                (width - self._kernel_size[1]) % self._stride[1])
            self._remainder['height'] = remainder_height
            self._remainder['width'] = remainder_width

        return self.layer(padded_x)

    @property
    def layer(self):
        return self._layer

    @property
    def padding_module(self):
        return self._padding_module

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    @property
    def remainder(self):
        if self._remainder is None:
            raise AttributeError(
                'Cannot access remainder of Conv2D module until .forward() has '
                'been called.')
        return self._remainder


class DeConv2D(Abstract2D):
    """Deconvolutional layer analogous torch.nn.ConvTranspose2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 stripping=0, output_padding=None,
                 output_padding_evaluated=None):
        """Construct deconvolutional layer.

        Args:
            in_channels: Int. Number of input channels.
            out_channels: Int. Number of output channels.
            kernel_size: Int or iterable of length 2. If int, represents the
                kernel size in each of the spatial dimensions.
            stride: Int or iterable of length 2. If int, represents the stride
                in each of the spatial dimensions.
            stripping: Iterable of non-negative ints of length 4 or 'SAME' or
                'VALID'.
                * If iterable, elements represent the width of the borders that
                    are stripped from to the left, right, top, and bottom of the
                    output respectively.
                * If 'SAME', stripping is automatically constructed to ensure
                    that the output spatial shape is the input spatial shape
                    divided by the stride.
                * If 'VALID', no stripping is used.
            output_padding: None or dictionary. If dictionary, must have  keys
                ['height', 'width']. The corresponding values indicate how much
                to zero-pad the output in the height and width spatial
                dimensions. This is useful if the user is trying to mirror a
                downsampling Conv2D layer which had some unused spatial
                remainder due to striding.
            output_padding_evaluated: None or callable returning Boolean. If
                None, the output_padding is used as is. If callable, checks to
                make sure the callable returns True before using the
                output_padding. This is a useful check and can help avoid gnarly
                bugs, so if output_padding is the remainder of a conv layer, be
                sure to feed in .space_evaluated of that layer as
                output_padding_evaluated.
        """
        super(DeConv2D, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(stripping, (list, tuple)):
            stripping = stripping
        elif isinstance(stripping, int):
            stripping = (stripping, stripping, stripping, stripping)
        elif stripping == 'VALID':
            stripping = (0, 0, 0, 0)
        elif stripping == 'SAME':
            top_stripping = kernel_size[0] // 2
            bottom_stripping = kernel_size[0] - top_stripping - 1
            left_stripping = kernel_size[1] // 2
            right_stripping = kernel_size[1] - left_stripping - 1
            stripping = (
                left_stripping, right_stripping, top_stripping, bottom_stripping
            )

        self._layer = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
        )

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._stripping = stripping

        if output_padding is None:
            self._output_padding = {'width': 0, 'height': 0}
        else:
            self._output_padding = output_padding
        if output_padding_evaluated is None:
            self._output_padding_evaluated = lambda: True
        else:
            self._output_padding_evaluated = output_padding_evaluated

    def _forward(self, x):
        """Forward application of the module.

        Args:
            x: Tensor with shape [batch, channels, height, width].

        Returns:
            out_x: Output tensor with shape
                [batch, self._out_channels, output_height, output_width].
        """
        if not self.space_evaluated():
            if not self._output_padding_evaluated():
                raise ValueError(
                    'Cannot compute the output padding before it is evaluated. '
                    'Check to make sure that the output_padding_evaluated '
                    'argument to the constructor of this instance is correct.')

            output_padding = (0, self._output_padding['width'],
                              0, self._output_padding['height'])
            self._output_padding_module = torch.nn.ZeroPad2d(output_padding)

        layer_x = self._layer(x)
        padded_layer_x = self._output_padding_module(layer_x)
        out_x = padded_layer_x[:, :, self._stripping[0]:-self._stripping[1],
                               self._stripping[2]:-self._stripping[3]]

        return out_x

    @property
    def layer(self):
        return self._layer

    @property
    def output_padding_module(self):
        return self._output_padding_module

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    @property
    def stripping(self):
        return self._stripping


class ReSize(Abstract2D):
    """Layer that resizes its input in the spatial dimensions."""

    def __init__(self, target, target_evaluated=None, mode='bilinear'):
        """Construct deconvolutional layer.

        Args:
            target: Dictionaty with keys ['height', 'width']. Corresponding
                values are the height and width to resize to.
            target_evaluated: None or callable returning Boolean. If None, the
                target is used as is. If callable, checks to make sure the
                callable returns True before using the output_padding. This is a
                useful check and can help avoid gnarly bugs, so if target is the
                remainder of a conv layer, be sure to feed in .space_evaluated
                of that layer as target_evaluated.
            mode: String. Model for the interpolation. See
                torch.nn.functional.interpolate for details.
        """
        super(ReSize, self).__init__()

        self._mode = mode
        self._target = target
        if target_evaluated is None:
            self._target_evaluated = lambda: True
        else:
            self._target_evaluated = target_evaluated

    def _forward(self, x):
        """Forward application of the module.

        Args:
            x: Tensor with shape [batch, channels, height, width].

        Returns:
            out_x: Output tensor with shape
                [batch, channels, self._target['height'], self._target['width']]
        """
        if not self.space_evaluated():
            if not self._target_evaluated():
                raise ValueError(
                    'Cannot compute the target size before the target it is '
                    'evaluated. Check to make sure that the target_evaluated '
                    'argument to the constructor of this instance is correct.')
            self._size = (self._target['height'], self._target['width'])

        out_x = torch.nn.functional.interpolate(
            x, size=self._size, mode=self._mode, align_corners=True)
        return out_x

    @property
    def mode(self):
        return self._mode

    @property
    def target(self):
        return self._target


class ReSizeAndConv(Abstract2D):
    """ReSize followed by same-padded stride 1 convolution.

    This is useful as an alternative to DeConv2D if you want to resize and apply
    a same-padded stride-1 convolution. This alternative is sometimes better
    than deconvolution because it helps avoid checkerboard artifacts and is
    easier to optimize. Be aware that it does use more memory though.
    """

    def __init__(self, conv2d, mode='bilinear'):
        """Create ReSizeAndConv module.

        Args:
            cnv2d: Instance of Conv2D. This will be used to compute channels,
                spatial shapes, and kernel size.
        """
        super(ReSizeAndConv, self).__init__()

        self._resize = get_resize_to_input(conv2d, mode=mode)
        self._conv = Conv2D(
            in_channels=conv2d.out_channels,
            out_channels=conv2d.in_channels,
            kernel_size=conv2d.kernel_size,
            stride=1,
            padding='SAME',
        )

    def _forward(self, x):
        """Forward application of the module."""
        x = self._resize.forward(x)
        x = self._conv.forward(x)
        return x

    @property
    def resize(self):
        return self._resize

    @property
    def conv(self):
        return self._conv


def get_deconv(conv2d):
    """Get symmetric deconvolution mirroring a Conv2D layer.

    Args:
        conv2d: Instance of Conv2d.

    Returns:
        transpose: Instance of DeConv2D.
    """
    transpose = DeConv2D(
        in_channels=conv2d.out_channels,
        out_channels=conv2d.in_channels,
        kernel_size=conv2d.kernel_size,
        stride=conv2d.stride,
        stripping=conv2d.padding,
        output_padding=conv2d.remainder,
        output_padding_evaluated=conv2d.space_evaluated,
    )
    return transpose


def get_resize_to_input(layer, mode='bilinear'):
    """Get resize to the input spatial shape of a layer.

    Args:
        layer: Instance of Abstract2D.

    Returns:
        resize: Instance of ReSize.
    """
    resize = ReSize(
        target=layer.input_spatial_shape,
        target_evaluated=layer.space_evaluated,
        mode=mode,
    )
    return resize


def get_resize_to_output(layer, mode='bilinear'):
    """Get resize to the output spatial shape of a layer.

    Args:
        layer: Instance of Abstract2D.

    Returns:
        resize: Instance of ReSize.
    """
    resize = ReSize(
        target=layer.output_spatial_shape,
        target_evaluated=layer.space_evaluated,
        mode=mode,
    )
    return resize
