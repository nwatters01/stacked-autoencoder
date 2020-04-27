"""Training script for autoencoder models.

This is forked from the cornet training script here:
https://github.com/dicarlolab/CORnet/blob/master/run.py
There are a few modifications from that script, such as Tensorboard logging of
scalars/images and no labels from the data loader.

To run, use a command like this on BrainTree:
'''bash
python run.py \
--data_path='/braintree/data2/active/common/imagenet_raw' \
--output_path='/braintree/home/nwatters/models/cornet/stacked_autoencoder/logs/test_0' \
--config='configs.layer_0' \
--batch_size=32 \
--epochs=10 \
train
'''

Be sure that you are running Python 3.6+
Also be sure that you have the following installed:
torch 1.2
torchvision 0.4
tensorboard
Those torch and torchvision versions are important and not the defaults if you
just pip install torch and torchvision!

Once running, you can monitor progress in Tensorboard. To do this, navigate to
the output_path specified in your lauch, and run the following:
'''bash
tensorboard --logdir=tensorboard
'''
This will launch tensorboard in a localhost on BrainTree. If you would like to
view that locally on your computer, you can run the following on your local
computer
ssh -N -L localhost:6006:localhost:6006 $username@$braintreehost.mit.edu
where $username is your username and $braintreehost is the braintree host you
are running on.
The 6006 port is Tensorboard's default, but feel free to use a different one.
"""


import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.utils.model_zoo
import torchvision
import importlib
import logging

from torch.utils import tensorboard

from PIL import Image
Image.warnings.simplefilter('ignore')

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--config', default='configs.stacked_ae_0',
                    help='which config to use')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')


FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)


if FLAGS.output_path is not None:
    if os.path.exists(FLAGS.output_path):
        # pass
        raise ValueError(
            'output_path {} exists, cannot overwrite. Please rerun with a '
            'different output_path.'.format(FLAGS.output_path))
    else:
        os.makedirs(FLAGS.output_path)
        summary_dir = os.path.join(FLAGS.output_path, 'tensorboard')
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)
        logging.info('summary_dir:  {}'.format(summary_dir))


def get_model():
    config_module = importlib.import_module(FLAGS.config)
    model = config_module.get_model()
    return model.cuda()


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.05,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=1,  # how often save model weigths
          save_model_secs=60 * 10  # how often save model (in sec)
          ):

    model = get_model()
    trainer = ImageNetTrain(model)
    validator = ImageNetVal(model)

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(1, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, (image, _) in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            image = image.cuda(non_blocking=True)
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    scalars_val = validator()
                    for k, v in scalars_val.items():
                        summary_writer.add_scalar(
                            'val_' + k, v, global_step=global_step)
                    results[validator.name] = scalars_val
                    trainer.model.train()

            if FLAGS.output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                trainer(frac_epoch, image)
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        record = trainer.scalars(image)
                        for k, v in record.items():
                            summary_writer.add_scalar(
                                'train_' + k, v, global_step=global_step)
                        record['data_load_dur'] = data_load_time
                        results[trainer.name] = record

                        # Log images
                        images = trainer.images(image)
                        for k, v in images.items():
                            summary_writer.add_image(
                                k, v, global_step=global_step)

            data_load_start = time.time()


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(128),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)
        return data_loader

    def scalars(self, inp):
        start = time.time()
        scalars = self.model.scalars(inp)
        scalars['learning_rate'] = self.lr.get_lr()[0]

        scalars['dur'] = time.time() - start
        return scalars

    def images(self, inp):
        images = self.model.images(inp)
        return images

    def __call__(self, frac_epoch, inp):
        self.lr.step(epoch=frac_epoch)
        loss = self.model.loss(inp)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'val_in_folders'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(150),
                torchvision.transforms.CenterCrop(128),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {k: 0. for k in self.model.scalar_keys}
        with torch.no_grad():
            for image, _ in tqdm.tqdm(self.data_loader, desc=self.name):
                inp = inp.cuda(non_blocking=True)
                scalars = self.model.scalars(image)
                for k, v in scalars.items():
                    record[k] += v

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
