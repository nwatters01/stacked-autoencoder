"""Model config to train lowest block of a stacked autoencoder."""

import nets
import stacked_autoencoder


def _get_block():
    encoder_layers = [
        nets.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=5,
            stride=2,
        ),
        nets.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=2,
        )
    ]

    block = stacked_autoencoder.AutoEncoderBlock(
        encoder_layers=encoder_layers,
        noise_stddev=0.2,
        activate_dec=False,
        decoder_type='resize',
    )

    return block


def get_model():
    block = _get_block()
    model = stacked_autoencoder.StackedAutoEncoder(
        block=block, prev_model=None)
    return model
