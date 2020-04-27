"""Stacked autoencoder class.

The main class in this file is StackedAutoEncoder. This model is an autoencoder
comprised of convolutional blocks. Each block autoencodes the representation
from the block below. The model is trained piecemeal, namely the lowest block is
trained first, then it is restored from snapshot to train the second block, etc.
"""

import image_utils
import nets
import torch
from torch import nn
from torch import distributions


class IdentityModel(object):
    """Dummy model used as the lowest module of a stacked autoencoder."""

    def full_encode(self, x_0):
        return x_0

    def full_decode(selfself, latent):
        return latent

    def images(self, image):
        return {}


def restore_model(model, checkpoint_path):
    """Returns a model restored from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])


class AutoEncoderBlock(nn.Module):
    """Block of a stacked autoencoder."""

    def __init__(self, encoder_layers, decoder_type='resize', noise_stddev=0.,
                 activation=None, activate_dec=False):
        """Construct autoencoder block.

        Args:
            encoder_layers: Iterable of layers. These layers should be instances
                of nets.Conv2D.
            decoder_type: String. 'resize' or 'deconv'. Which decoder type to
                use.
            noise_stddev: Float. Stddev of the latent space noise.
            Activation: None or activation module. If None, defaults to
                torch.nn.Sigmoid().
            activate_dec: Boolean. Whether to apply activation function to
                decoder.
        """
        super(AutoEncoderBlock, self).__init__()
        self._activate_dec = activate_dec
        self._encoder_layers = encoder_layers
        if decoder_type == 'resize':
            self._decoder_layers = tuple([
                nets.ReSizeAndConv(conv2d=conv2d)
                for conv2d in self._encoder_layers[::-1]
            ])
        elif decoder_type == 'deconv':
            self._decoder_layers = tuple([
                nets.get_deconv(conv2d)
                for conv2d in self._encoder_layers[::-1]
            ])
        else:
            raise ValueError('Invalid decoder_tpye {}'.format(decoder_type))

        for i, enc_module in enumerate(self._encoder_layers):
            self.add_module("_enc_module_" + str(i), enc_module)
        for i, dec_module in enumerate(self._decoder_layers):
            self.add_module("_dec_module_" + str(i), dec_module)

        self._noise = distributions.normal.Normal(0., noise_stddev)

        if activation is None:
            activation = nn.Sigmoid()
        self._activation = activation

    def forward(self, x, noise=True):
        """Autoencode input x, optionally with latent noise."""
        z = self.encode(x)
        if noise:
            z = z + self._noise.sample()
        recons = self.decode(z)
        return recons

    def encode(self, x):
        """Encode from input space x to latent space."""
        for conv2d in self._encoder_layers:
            x = conv2d(x)
            x = self._activation(x)
        return x

    def decode(self, x):
        """Decode from latent space x to input space."""
        x = self._decoder_layers[0](x)
        for conv2d in self._decoder_layers[1:]:
            x = self._activation(x)
            x = conv2d(x)
        if self._activate_dec:
            x = self._activation(x)
        return x


class StackedAutoEncoder(nn.Module):
    """Stacked autoencoder model."""

    def __init__(self, block, prev_model=None):
        """Construct stacked autoencoder model.

        Args:
            block: Instance of AutoEncoderBlock. The current block to train.
            prev_model: None or instance of StackedAutoEncoder. If None,
                defaults  to IdentityModel. This model is the model below the
                current block. Namely, the block is trained to autoencode the
                latent space of prev_model.
        """
        super(StackedAutoEncoder, self).__init__()
        self.add_module("_block", block)
        if prev_model is None:
            self._prev_model = IdentityModel()
        else:
            self.add_module("_prev_model", prev_model)
            for param in self._prev_model.parameters():
                param.requires_grad = False
        self._loss = nn.MSELoss().cuda()

    def forward(self, image):
        """Forward pass on an image.

        Args:
            image: Tensor of shape [batch, channels, height, width]. Input
                image.

        Returns:
            prev_latent: 4-tensor. Latent space of self._prev_model.
            recons: Current block's reconstruction of prev_latent.
        """
        prev_latent = self._prev_model.full_encode(image)
        recons = self._block.forward(prev_latent.detach())
        return prev_latent, recons

    def full_encode(self, image):
        """Encode from image, returning current latent space."""
        prev_latent = self._prev_model.full_encode(image)
        return self._block.encode(prev_latent)

    def full_decode(self, latent):
        """Decode from latent, return reconstructed image."""
        recons = self._block.decode(latent)
        return self._prev_model.full_decode(recons)

    def loss(self, image):
        """Compute model loss to be optimized."""
        prev_latent, recons = self.forward(image)
        return self._loss(prev_latent.detach(), recons)

    def scalars(self, image):
        """Get dictionary of scalars for logging."""
        prev_latent, recons = self.forward(image)
        scalars = {
            'loss': self._loss(prev_latent, recons),
            'mean_prev_latent': torch.mean(prev_latent),
            'mean_recons': torch.mean(recons),
            'std_prev_latent': torch.std(prev_latent),
            'std_recons': torch.std(recons),
        }
        return scalars

    @property
    def scalar_keys(self):
        scalar_keys = [
            'loss', 'mean_input', 'mean_recons', 'std_input', 'std_recons']
        return scalar_keys

    def images(self, image):
        """Get dictionary of numpy images for logging."""
        decoding = self.full_decode(self.full_encode(image))
        images = {
            'input': image,
            'recons': decoding,
            'diff': image - decoding,
        }
        images = {
            k: image_utils.images_to_grid(
                image_utils.normalize_0_1(v.detach().cpu().numpy()),
                padding_value=1.)
            for k, v in images.items()
        }
        prev_images = self._prev_model.images(image)
        images.update(
            {'prev_' + k: v for k, v in prev_images.items()}
        )
        return images

