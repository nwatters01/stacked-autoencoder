"""Model config to train fourth block of a stacked autoencoder."""

from configs import layer_2
import nets
import stacked_autoencoder


def _get_block():
    encoder_layers = [
        nets.Conv2D(
            in_channels=128,
            out_channels=192,
            kernel_size=5,
            stride=1,
        ),
        nets.Conv2D(
            in_channels=192,
            out_channels=192,
            kernel_size=5,
            stride=2,
        )
    ]

    block = stacked_autoencoder.AutoEncoderBlock(
        encoder_layers=encoder_layers,
        noise_stddev=0.2,
        activate_dec=True,
        decoder_type='resize',
    )

    return block


def _get_prev_model():

    checkpoint_path = (
        '/braintree/home/nwatters/models/cornet/sae_logs/noise_0_layer_2/0/'
        'latest_checkpoint.pth.tar'
    )
    prev_model = layer_2.get_model()
    stacked_autoencoder.restore_model(prev_model,
                                      checkpoint_path=checkpoint_path)

    return prev_model


def get_model():
    block = _get_block()
    prev_model = _get_prev_model()
    model = stacked_autoencoder.StackedAutoEncoder(
        block=block, prev_model=prev_model)
    return model
