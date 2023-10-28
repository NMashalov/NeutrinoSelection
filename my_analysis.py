import h5py as h5
import tensorflow as tf
import os
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

from ds_making import make_dataset


## GPU on
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#    print(gpu)
#    tf.config.experimental.set_memory_growth(gpu, True)

### E and S analysis functions

def make_preds(model, regime, path_to_h5, Shape=(None, 6), bs=2048):
    # path_to_model = '../trained_models/' + mn+'/'+model_checkpoint
    # model = tf.keras.models.load_model(path_to_model, compile=False)
    with h5.File(path_to_h5, 'r') as hf:
        L = len(hf[regime + '/ev_ids_corr/data'])
    dataset = make_dataset(path_to_h5, regime=regime, batch_size=bs, shape=Shape)
    proba_nn = model.predict(dataset, steps=L // bs)

    # making dir for preds if necessary
    try:
        os.makedirs('analysis/predictions')
        print('directory for preds is created')
    except:
        print('directory for preds already exists')
    mn = model._name
    path_to_preds = './predictions/preds_' + mn + '_' + regime
    np.save(path_to_preds, proba_nn)

    return path_to_preds


def load_preds_and_labels(model_name, regime, path_to_h5):
    proba_nn = np.load('./predictions/preds_' + model_name + '_' + regime + '.npy')
    with h5.File(path_to_h5, 'r') as hf:
        L = proba_nn.shape[0]
        labels = np.zeros((L, 1))
        ids = hf[regime + '/ev_ids_corr/data'][0:L]  # id of event - starting with 'nu' or 'mu'
        # ids = np.array([i[0] for i in ids]).reshape(ids.shape[0],1)
        ids_mu = np.where(np.char.startswith(ids, b'mu'))[0]
        ids_nuatm = np.where(np.char.startswith(ids, b'nuatm'))[0]
        ids_nue2 = np.where(np.char.startswith(ids, b'nue2'))[0]
        labels[ids_mu] = 0
        labels[ids_nuatm] = 1
        labels[ids_nue2] = 2
    return proba_nn[:, 1], labels[:, 0]


def separate_preds(preds, labels):
    idxs_mu = np.where(labels == 0)[0]
    idxs_nuatm = np.where(labels == 1)[0]
    idxs_nue2 = np.where(labels == 2)[0]
    preds_mu = preds[idxs_mu]
    preds_nuatm = preds[idxs_nuatm]
    preds_nue2 = preds[idxs_nue2]
    return preds_mu, preds_nuatm, preds_nue2


def get_positive(proba, tr_start=0., tr_end=1., N_points=5000):
    tr_arr = np.linspace(tr_start, tr_end, N_points)
    Pos = []
    for tr in tr_arr:
        Pos.append(np.sum(np.where(proba >= tr, 1, 0)))
    return np.array(Pos)


def get_MuNu_positive(preds_mu, preds_nu, tr_start=0., tr_end=1., N_points=5000):
    tr_arr = np.linspace(tr_start, tr_end, N_points)
    MuPos, NuPos = [], []
    for tr in tr_arr:
        MuPos.append(np.where(preds_mu >= tr)[0].shape[0])
        NuPos.append(np.where(preds_nu >= tr)[0].shape[0])
    return np.array(MuPos), np.array(NuPos)


def plot_SE(S, E, tr_start=0., tr_end=1.):
    x_log_start = tr_start + 0.8 * (tr_end - tr_start)
    x_tr = np.linspace(tr_start, tr_end, S.shape[0])
    fig, (ax1, ax0) = plt.subplots(1, 2, figsize=(16, 7))
    i_start = int(x_log_start * S.shape[0])
    # log scale plot
    ax0.plot(x_tr[i_start:], E[i_start:], label='Exposure')
    ax0.plot(x_tr[i_start:], S[i_start:], label='Suppression')
    ax0.set_title("E and S vs classification threshold", fontsize=14)
    ax0.set_xlabel("Threshold", fontsize=14)
    ax0.set_ylabel("E and S values", fontsize=14)
    ax0.set_yscale("log")
    ax0.legend(fontsize=12, loc=0)
    ax0.grid()
    # normal plot
    ax1.plot(x_tr, E, label='Exposure')
    ax1.plot(x_tr, S, label='Suppression')
    ax1.set_title("E and S vs classification threshold", fontsize=14)
    ax1.set_xlabel("Threshold", fontsize=14)
    ax1.set_ylabel("E and S values", fontsize=14)
    ax1.legend(fontsize=12, loc=0)
    ax1.grid()
    # plt.savefig('figures/'+model_name+'_'+regime+'_'+str(x_start/1000)+'_'+str(x_end/1000)+'.png')
    plt.show()
    return fig
    # plt.close(fig)


def get_NuFromNN(MuPos_test, NuPos_test, Pos_val, N_val=None, N_mu_test=None, N_nu_test=None, alpha=1. - 0.68):
    ### Формула потока
    S = MuPos_test / N_mu_test
    E = NuPos_test / N_nu_test
    NuFromNN = (Pos_val - S * N_val) / (E - S)

    ### Оценка ошибки
    sigma_S, sigma_E = [], []

    # Считаем погрешность S                 
    n = N_mu_test  # MuPos_test[0]
    for k in MuPos_test:
        low, up = beta.ppf([alpha / 2, 1 - alpha / 2],
                           [k, k + 1],
                           [n - k + 1, n - k])
        low = np.nan_to_num(low)
        sigma_S.append((up - low) / 2)  # (max(k/n - low, up - k/n))

    # Считаем погрешность E                 
    n = N_nu_test  # NuPos_test[0]
    for k in NuPos_test:
        low, up = beta.ppf([alpha / 2, 1 - alpha / 2],
                           [k, k + 1],
                           [n - k + 1, n - k])
        up = np.nan_to_num(up, nan=1.0)
        sigma_E.append((up - low) / 2)

    # Считаем погрешность формулы для N                 
    sigma_S, sigma_E = np.array(sigma_S), np.array(sigma_E)
    S1 = (Pos_val - S * N_val) ** 2 / (E - S) ** 4 * sigma_E ** 2
    S2 = (Pos_val - E * N_val) ** 2 / (E - S) ** 4 * sigma_S ** 2
    S3 = 0  # ((1-2*S)*Pos_val+Pos_val[0]*S**2)/(E-S)**2
    sigma_N = np.sqrt(S1 + S2 + S3)

    return NuFromNN, sigma_N, S1, S2, S3


def plot_error_and_flux(NuFromNN, sigma_N, true_Nu, tr_start=0., tr_end=1.):
    x_tr = np.linspace(tr_start, tr_end, NuFromNN.shape[0])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 7))
    # error plot
    # i_start, i_end = int(tr_start*NuFromNN.shape[0]), int(tr_end*NuFromNN.shape[0])
    ax0.plot(x_tr, sigma_N, label='Абс. ошибка числа нейтрино', color='red')
    ax0.set_title("The error of neutrino flux", fontsize=10)
    ax0.set_xlabel("treshold", fontsize=10)
    ax0.set_ylabel("Error value", fontsize=10)
    ax0.legend(fontsize=8, loc=0)
    ax0.grid()
    # flux plot
    ax1.plot(x_tr, NuFromNN, label='Number of nu given by the formula')
    ax1.plot(x_tr, (true_Nu * np.ones(x_tr.shape[0])), color='green', label="True number of nu")
    ax1.plot(x_tr, (true_Nu * np.ones(x_tr.shape[0]) + sigma_N), '--', color='red', label="Limits of error")
    ax1.plot(x_tr, (true_Nu * np.ones(x_tr.shape[0]) - sigma_N), '--', color='red')
    ax1.set_title("The evaluation of neutrino flux", fontsize=10)
    ax1.set_xlabel("treshold", fontsize=10)
    ax1.set_ylabel("Nu number", fontsize=10)
    ax1.legend(fontsize=8, loc=0)
    ax1.grid()
    # plt.savefig('Figures_flux/'+model_name+'_'+regime+'_'+str(x_start/1000)+'_'+str(x_end/1000)+'.png')
    plt.show()
    return fig
