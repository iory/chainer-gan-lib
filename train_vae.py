import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from common.dataset import Cifar10Dataset
from common.dataset import ImagenetDataset
from common.evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID
from common.record import record_setting
import common.net

def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--algorithm', '-a', type=str, default="dcgan", help='GAN algorithm')
parser.add_argument('--architecture', type=str, default="dcgan", help='Network architecture')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
parser.add_argument('--evaluation_interval', type=int, default=10000, help='Interval of evaluation')
parser.add_argument('--display_interval', type=int, default=100, help='Interval of displaying log to console')
parser.add_argument('--n_dis', type=int, default=5, help='number of discriminator update per generator update')
parser.add_argument('--gamma', type=float, default=0.5, help='hyperparameter gamma')
parser.add_argument('--lam', type=float, default=10, help='gradient penalty')
parser.add_argument('--adam_alpha', type=float, default=0.0001, help='alpha in Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.5, help='beta1 in Adam optimizer')
parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the discriminator (for cramer GAN)')

args = parser.parse_args()
record_setting(args.out)
report_keys = ["loss_enc", "loss_dis", "loss_gen", "inception_mean", "inception_std", "FID"]

# Set up dataset
# train_dataset = Cifar10Dataset()
train_dataset = ImagenetDataset()
train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

# Setup algorithm specific networks and updaters
models = []
opts = {}
updater_args = {
    "iterator": {'main': train_iter},
    "device": args.gpu
}


from vaegan.updater import Updater
encoder = common.net.VAEEncoder(size=256)
generator = common.net.DCGANGenerator(n_hidden=100, z_distribution="normal")
discriminator = common.net.DCGANDiscriminator()
models = [encoder, generator, discriminator]


if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    print("use gpu {}".format(args.gpu))
    for m in models:
        m.to_gpu()

# Set up optimizers
opts["opt_enc"] = make_optimizer(encoder, args.adam_alpha, args.adam_beta1, args.adam_beta2)
opts["opt_enc"].add_hook(chainer.optimizer.WeightDecay(0.00001))
opts["opt_gen"] = make_optimizer(generator, args.adam_alpha, args.adam_beta1, args.adam_beta2)
opts["opt_gen"].add_hook(chainer.optimizer.WeightDecay(0.00001))
opts["opt_dis"] = make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2)
opts["opt_dis"].add_hook(chainer.optimizer.WeightDecay(0.00001))

updater_args["optimizer"] = opts
updater_args["models"] = models

# Set up updater and trainer
updater = Updater(**updater_args)
trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

# Set up logging
for m in models:
    trainer.extend(extensions.snapshot_object(
        m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
trainer.extend(extensions.LogReport(keys=report_keys,
                                    trigger=(args.display_interval, 'iteration')))
trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
trainer.extend(sample_generate(generator, args.out), trigger=(args.evaluation_interval, 'iteration'),
               priority=extension.PRIORITY_WRITER)
trainer.extend(sample_generate_light(generator, args.out), trigger=(args.evaluation_interval // 10, 'iteration'),
               priority=extension.PRIORITY_WRITER)
trainer.extend(calc_inception(generator), trigger=(args.evaluation_interval, 'iteration'),
               priority=extension.PRIORITY_WRITER)
trainer.extend(calc_FID(generator), trigger=(args.evaluation_interval, 'iteration'),
               priority=extension.PRIORITY_WRITER)
trainer.extend(extensions.ProgressBar(update_interval=10))

# Run the training
trainer.run()
