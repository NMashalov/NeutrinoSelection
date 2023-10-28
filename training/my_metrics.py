# Metric to see TP and FP for multiple values of treshold
import tensorflow as tf
import numpy as np


class Expos_on_Suppr(tf.keras.metrics.Metric):
    def __init__(self, tr_list=None, max_suppr_value=1e-5, name='Expos_on_Suppr', tr_start=0.9, num_of_points=100000, **kwargs):
        super(Expos_on_Suppr, self).__init__(name=name, **kwargs)
        if tr_list is None:
            tr_list = list(np.linspace(tr_start, 1., num_of_points))
        self.i_best = -1
        self.idxs = []
        self.tr_list = tr_list
        self.suppr = [1.] * len(self.tr_list)
        self.expos_on_suppr = -1.
        self.max_suppr_value = max_suppr_value
        self.fp = tf.keras.metrics.FalsePositives(
            thresholds=tr_list)
        self.tp = tf.keras.metrics.TruePositives(
            thresholds=tr_list)
        self.tn = tf.keras.metrics.TrueNegatives(
            thresholds=tr_list)
        self.fn = tf.keras.metrics.FalseNegatives(
            thresholds=tr_list)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        self.fp.update_state(y_true, y_pred)
        self.tp.update_state(y_true, y_pred)
        self.tn.update_state(y_true, y_pred)
        self.fn.update_state(y_true, y_pred)
        self.suppr = tf.divide(self.fp.result(), tf.add(self.fp.result(), self.tn.result()))

        self.idxs = tf.where(tf.less(self.suppr, self.max_suppr_value))
        self.i_best = self.idxs[0][0]
        if tf.equal(self.suppr[self.i_best],0.):
            self.expos_on_suppr = -1.
        else:
            self.expos_on_suppr = tf.divide(self.tp.result()[self.i_best],
                                            tf.add(self.tp.result()[self.i_best],
                                                   self.fn.result()[self.i_best]))

    def reset_state(self):
        self.fp.reset_state()
        self.tp.reset_state()
        self.tn.reset_state()
        self.fn.reset_state()
        #self.suppr = [1.] * len(self.tr_list)
        #self.expos_on_suppr = -1.
        #self.idxs = []
        #self.i_best = -1

    def result(self):
        return self.expos_on_suppr
