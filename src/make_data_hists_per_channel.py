import numpy as np
from customs.plot_hists_per_channel import plot_hists
from customs.info_data_flat import info_Baikal_HDF5_flat


h5_name = 'baikal_multi_0523_flat_pureMC_h5s2_norm.h5'
path_to_data = './data/'
path_to_h5 = path_to_data + h5_name
info = info_Baikal_HDF5_flat(h5_name=h5_name,
                             path_to_data=path_to_data)

# Create distribution figs for flat data
for regime in ['test']:
    if regime == 'train':
        size = int(2e4)
    else:
        size = -1

    data_mu, len_mu = info.get_flat_data_samples(size, particle='mu', regime=regime)
    data_mu[np.where(data_mu[:, 0] > 100)[0], 0] = 100
    f_mu = plot_hists(data_mu, len_mu,
                      title="EAS distributions FLAT " + regime)

    data_nu_atm, len_nu_atm = info.get_flat_data_samples(size, particle='nu_atm', regime=regime)
    data_nu_atm[np.where(data_nu_atm[:, 0] > 100)[0], 0] = 100
    f_nu_atm = plot_hists(data_nu_atm, len_nu_atm,
                          title="NuAtm distributions FLAT " + regime)

    data_nu_e2, len_nu_e2 = info.get_flat_data_samples(size, particle='nu_e2', regime=regime)
    data_nu_e2[np.where(data_nu_e2[:, 0] > 100)[0], 0] = 100
    f_nu_e2 = plot_hists(data_nu_e2, len_nu_e2,
                         title="NuE2 distributions FLAT " + regime)

    f_nu = plot_hists(np.concatenate([data_nu_atm, data_nu_e2], axis=0),
                      np.concatenate([len_nu_atm, len_nu_e2], axis=0),
                      title="NuAll distributions FLAT " + regime)