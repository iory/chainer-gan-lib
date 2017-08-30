import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.enc, self.gen, self.dis = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        enc_optimizer = self.get_optimizer('opt_enc')
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp

        batch = self.get_iterator('main').next()
        batch_size = len(batch)
        x = []
        for i in range(batch_size):
            x.append(np.asarray(batch[i]).astype("f"))
        x_real = chainer.Variable(xp.asarray(x))

        # encode
        z0, mean, var = self.enc(x_real)
        x0 = self.gen(z0)
        y0, l0 = self.dis(x0)
        loss_enc = F.gaussian_kl_divergence(mean, var) / float(l0.data.size)
        loss_gen = 0
        loss_gen = F.softmax_cross_entropy(y0, chainer.Variable(xp.zeros(batch_size).astype(np.int32)))
        loss_dis = F.softmax_cross_entropy(y0, chainer.Variable(xp.ones(batch_size).astype(np.int32)))
        # train generator
        z1 = chainer.Variable(xp.random.normal(0, 1, (batch_size, self.enc.latent_size)).astype(np.float32))
        x1 = self.gen(z1)
        y1, l1 = self.dis(x1)
        loss_gen += F.softmax_cross_entropy(y1, chainer.Variable(xp.zeros(batch_size).astype(np.int32)))
        loss_dis += F.softmax_cross_entropy(y1, chainer.Variable(xp.ones(batch_size).astype(np.int32)))
        # train discriminator
        y2, l2 = self.dis(chainer.Variable(xp.asarray(x)))
        loss_enc += F.mean_squared_error(l0, l2)
        loss_gen += 0.1 * F.mean_squared_error(l0, l2)
        loss_dis += F.softmax_cross_entropy(y2, chainer.Variable(xp.zeros(batch_size).astype(np.int32)))

        self.enc.cleargrads()
        loss_enc.backward()
        enc_optimizer.update()

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_enc': loss_enc})
        chainer.reporter.report({'loss_gen': loss_gen})
        chainer.reporter.report({'loss_dis': loss_dis})
