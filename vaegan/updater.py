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

        # # encode
        # z0, mean, var = self.enc(x_real)
        # x0 = self.gen(z0)
        # y0, l0 = self.dis(x0)

        # loss_enc = F.gaussian_kl_divergence(mean, var) / batch_size
        # loss_gen = F.softmax_cross_entropy(y0, chainer.Variable(xp.zeros(batch_size).astype(np.int32)))
        # loss_dis = F.softmax_cross_entropy(y0, chainer.Variable(xp.ones(batch_size).astype(np.int32)))

        # # train generator
        # z1 = chainer.Variable(xp.asarray(self.gen.make_hidden(batch_size)))
        # x1 = self.gen(z1)
        # y1, l1 = self.dis(x1)
        # loss_gen += F.softmax_cross_entropy(y1, chainer.Variable(xp.zeros(batch_size).astype(np.int32)))
        # loss_dis += F.softmax_cross_entropy(y1, chainer.Variable(xp.ones(batch_size).astype(np.int32)))
        # # train discriminator
        # y2, l2 = self.dis(chainer.Variable(xp.asarray(x)))
        # loss_enc += F.mean_squared_error(l0, l2)
        # loss_gen += F.mean_squared_error(l0, l2) * l0.data.shape[2] * l0.data.shape[3]
        # loss_dis += F.softmax_cross_entropy(y2, chainer.Variable(xp.zeros(batch_size).astype(np.int32)))

        # real image
        y_real, l2_real = self.dis(x_real)
        L_dis_real = F.softmax_cross_entropy(y_real,
                                             chainer.Variable(xp.zeros(batch_size).astype(np.int32)))

        # fake image from random noize
        z = chainer.Variable(xp.asarray(self.gen.make_hidden(batch_size)))
        x_fake = self.gen(z)
        y_fake, _ = self.dis(x_fake)
        L_gen_fake = F.softmax_cross_entropy(y_fake,
                                             chainer.Variable(xp.zeros(batch_size).astype(np.int32)))
        L_dis_fake = F.softmax_cross_entropy(y_fake,
                                             chainer.Variable(xp.ones(batch_size).astype(np.int32)))

        # fake image from reconstruction
        z_rec, mu_z, ln_var_z = self.enc(x_real)
        x_rec = self.gen(z_rec)
        y_rec, l2_rec  = self.dis(x_rec)

        L_prior = F.gaussian_kl_divergence(mu_z, ln_var_z) / batch_size
        L_gen_rec = F.softmax_cross_entropy(y_rec, chainer.Variable(xp.zeros(batch_size).astype(np.int32)))
        L_dis_rec = F.softmax_cross_entropy(y_rec, chainer.Variable(xp.ones(batch_size).astype(np.int32)))

        L_rec = F.mean_squared_error(l2_real, l2_rec) * l2_real.data.shape[2] * l2_real.data.shape[3]

        # calc loss
        loss_dis = L_dis_real + 0.5 * L_dis_fake + 0.5 * L_dis_rec
        loss_enc = L_prior + L_rec
        loss_gen = L_gen_fake + L_gen_rec + L_rec


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
