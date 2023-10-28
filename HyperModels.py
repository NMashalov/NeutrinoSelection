from customs.losses import focal_loss
from customs.my_metrics import Expos_on_Suppr
from nn import *
import tensorflow as tf
import keras_tuner as kt


class LstmHyperModel(kt.HyperModel):

    def __init__(self, batch_size=256, lr_opt=True, lr=0.005, decay_rate=0.99, N=2):
        self.bs = batch_size
        self.lr_opt = lr_opt
        self.lr = lr
        self.decay_rate = decay_rate
        self.N = N

    def customise_HP(self, hp):
        if self.lr_opt:
            self.lr = hp.Float('lr_i', 0.0025, 0.04, step=0.0025, sampling="linear")
            self.decay_rate = hp.Choice('decay_rate', [0.8, 0.9, 0.95])
            self.u_list = [16, 16, 16, 16]
            self.act = 'tanh'
            self.rec_act = 'sigmoid'
            self.merge_mode = 'mul'
            self.is_mask = True
            self.weight_in_loss = 10.
        else:
            self.u_list = []
            for i in range(self.N):
                self.u_list.append(hp.Choice('u' + str(i + 1), [int(4), int(16), int(64)]))
            self.act = hp.Choice('act', ['tanh', 'sigmoid'])
            self.rec_act = hp.Choice('rec_act', ['tanh', 'sigmoid'])
            self.merge_mode = hp.Choice('merge_mode', ['concat', 'mul'])
            self.is_mask = True  # hp.Choice('mask', [True, False])
            self.weight_in_loss = hp.Choice('loss_w', [2., 5., 10., 15.])

    def build(self, hp, Shape=(None, 6)):
        self.customise_HP(hp)
        model = nn_rnn_model(Shape, self.N, self.u_list, self.act, self.rec_act, self.merge_mode, self.is_mask)
        print(model.summary())
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=1000,
            decay_rate=self.decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                             amsgrad=False, name='Adam')

        model.compile(optimizer=optimizer, loss=focal_loss(2., 2., self.weight_in_loss, 1.),
                      metrics=[Expos_on_Suppr(),
                               'accuracy'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=self.bs,
                         verbose=True, **kwargs)
