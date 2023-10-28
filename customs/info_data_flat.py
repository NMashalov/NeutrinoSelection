import numpy as np
import h5py as h5

# Class to work woth flat data
class info_Baikal_HDF5_flat:
    def __init__(self, h5_name, path_to_data='./data/'):
        self.h5_name = h5_name
        self.path = path_to_data + h5_name
        # self.nums_in_train, self.nums_in_test, self.nums_in_val = None, None, None
        self.nums_dict = {'train': None, 'test': None, 'val': None}
        self.idxs_dict = {'train': None, 'test': None, 'val': None}

    def collect_info(self, regime):
        with h5.File(self.path, 'r') as hf:
            ids = hf[regime + '/ev_ids_corr/data']
            ids_mu = np.where(np.char.startswith(ids, b'mu'), 1, 0)
            ids_nuatm = np.where(np.char.startswith(ids, b'nuatm'), 1, 0)
            ids_nue2 = np.where(np.char.startswith(ids, b'nue2'), 1, 0)
            print('ids_collected')
            num_mu = len(ids_mu)
            print('mu_collected')
            num_atm_nu = len(ids_nuatm)
            print('nu_atm_collected')
            num_e2_nu = len(ids_nue2)
            print('nu_e2_collected')
            self.nums_dict[regime] = {'mu': num_mu, 'nu_atm': num_atm_nu, 'nu_e2': num_e2_nu}
            self.idxs_dict[regime] = {'mu': ids_mu, 'nu_atm': ids_nuatm, 'nu_e2': ids_nue2}

    def get_flat_data_samples(self, size, particle='mu', regime='val', return_norm=False):
        r = regime
        with h5.File(self.path, 'r') as hf:
            mean = hf['norm_param/mean']
            std = hf['norm_param/std']
            if self.idxs_dict[r] == None:
                self.collect_info(r)
            idxs = self.idxs_dict[r][particle]
            idxs = np.where(idxs == 1)[0][0:size]
            len_ev = np.diff(hf[r + '/ev_starts/data'])[idxs]
            if r == 'train':
                ev_starts = hf[r + '/ev_starts/data'][idxs]
                ev_ends = hf[r + '/ev_starts/data'][idxs + 1]
                data = []  # np.zeros((0,5))
                for s, e in zip(ev_starts, ev_ends):
                    data.append(hf[r + '/data/data'][s:e])
                data = np.concatenate(data, axis=0)
            else:
                start = hf[r + '/ev_starts/data'][idxs[0]]
                end = hf[r + '/ev_starts/data'][idxs[-1] + 1]
                data = hf[r + '/data/data'][start:end]

            if not return_norm:
                data = data * std + mean

            return data, len_ev