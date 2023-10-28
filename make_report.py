import h5py as h5    
from my_analysis import *
import numpy as np
from IPython.display import clear_output

# data and model's names
data_names = [n for n in os.listdir('data/') if n.endswith('.h5')]
for i,h5n in enumerate(data_names):
    print(str(i+1),". "+h5n)
i = int(input("Which dataset do you want to use? Print it's number! \n"))
name = data_names[i-1]
path_to_h5 = '../data/' + name

model_names = [n for n in os.listdir('trained_models/') if not n.startswith('logs') and not n.startswith('.')]
for i,mn in enumerate(model_names):
    print(str(i+1),". "+mn)
i = int(input("Which model do you want to choose? Print it's number! \n"))
model_name = model_names[i-1]

model = None
trigger = input("Do you want to create new predictions for 'val' regime? Type only 'y' or 'n': \n")
if trigger == 'y':
    if model is None:
        path_to_model = '../trained_models/' + model_name +'/'+'best' #Change to best later!
        model = tf.keras.models.load_model(path_to_model, compile=False)
        model._name = model_name
        clear_output(wait=False)
    _ = make_preds(model, 'val', path_to_h5, bs = 512)
elif trigger == 'n':
    pass
else:
    print("Your input is incorrect. Preds will not be recreated.")

# create report dir
path_to_report = './preds_analysis/report_' + model_name + '_' + name
try:
    os.makedirs(path_to_report)
    print('directory for report is created')
except:
    print('directory for report already exists')

proba_val, labels_val = load_preds_and_labels(model_name, 'val', path_to_h5)
print("All predictions are loaded!")

### SOMETHING GLOBAL ###
N_points = 10000
tr_start, tr_end = 0., 0.9999

# Get MuPos and NuPos for 'val'
preds_mu_val, preds_nuatm_val, preds_nue2_val = separate_preds(proba_val, labels_val)
preds_nu_val = np.concatenate([preds_nuatm_val,preds_nue2_val])
print("Predictions are separated to Mu -- Nu.")
MuPos_val, NuPos_val = get_MuNu_positive(preds_mu_val, preds_nu_val, tr_start = tr_start, tr_end = tr_end, N_points = N_points)
Pos_val = MuPos_val + NuPos_val
print("Numbers of positives Mu and Nu are calculated for each treshold!")
np.save(path_to_report+"/MuPos_val.npy", MuPos_val)
np.save(path_to_report+"/NuPos_val.npy", NuPos_val)
print(tr_start, tr_end, N_points)

# Get S and E
S_val = MuPos_val/preds_mu_val.shape[0]
E_val = NuPos_val/preds_nu_val.shape[0]
i_crit = np.argwhere(S_val<=1e-6)[0]
report_file = open(path_to_report+"/info_ES.txt", "w")
report_file.write("Critical treshold = " + str(i_crit/S_val.shape[0] * (tr_end-tr_start)) + '\n')
report_file.write("Level of suppression = " + str(S_val[i_crit]) + '\n')
report_file.write("Level of exposition = " + str(E_val[i_crit]) + '\n')
report_file.close()
print("Critical treshold =",i_crit/S_val.shape[0] * (tr_end-tr_start))
print("Level of suppression =", S_val[i_crit])
print("Level of exposition =", E_val[i_crit])

# Plot S and E
fig_SE = plot_SE(S_val, E_val, tr_start=tr_start, tr_end=tr_end)
print("Picture of S and E is created!")
fig_SE.savefig(path_to_report+"/fig_SE.png")

### separate val data to new test and new val
true_Nu = 30 #desired number of nu in flow
true_Mu = int(3e6) #desired number of mu in flow
start_Mu, start_Nu = int(3e6), 0 #where to start slices
preds_mu_test = np.concatenate([preds_mu_val[start_Mu+true_Mu:],preds_mu_val[0:start_Mu]], axis = 0)
preds_nu_test = np.concatenate([preds_nu_val[start_Nu+true_Nu:],preds_nu_val[0:start_Nu]], axis = 0)
report_file = open(path_to_report+"/info_flux.txt", "w")
report_file.write(f"Mu number = {true_Mu}\nNu number = {true_Nu}\nStarts of slices = {start_Mu}, {start_Nu}.")
report_file.close()
print("Test preds for neutrino flux estimation are extracted!")
preds_mu_val_new, preds_nu_val_new = preds_mu_val[start_Mu:start_Mu+true_Mu], preds_nu_val[start_Nu:start_Nu+true_Nu]
print("Val preds for neutrino flux estimation are extracted!")
MuPos_test, NuPos_test = get_MuNu_positive(preds_mu_test, preds_nu_test, tr_start = tr_start, tr_end = tr_end, N_points = N_points)
print("Numbers of positives Mu and Nu in 'test' are calculated for each treshold!")
MuPos_val_new, NuPos_val_new = get_MuNu_positive(preds_mu_val_new, preds_nu_val_new, tr_start = tr_start, tr_end = tr_end, N_points = N_points)
print("Numbers of positives Mu and Nu in 'val' are calculated for each treshold!")

# calculate Nu flux and its error
Pos_val_new = NuPos_val_new + MuPos_val_new #total number of events depending on treshold
N, sigma_N, S1, S2, S3 = get_NuFromNN(MuPos_test, NuPos_test, Pos_val_new, 
                                      N_val = preds_mu_val_new.shape[0]+preds_nu_val_new.shape[0], 
                                      N_mu_test = preds_mu_test.shape[0], 
                                      N_nu_test = preds_nu_test.shape[0]) #evaluate flux and its error
print("Number of neutrino in new 'val' dataset is calculated with its error!")
print('Mu number =', true_Mu)
print('True Nu number =', true_Nu)
cut = 0.
i = int(len(N)*cut)
fig_flux = plot_error_and_flux(N[i:], sigma_N[i:], true_Nu = true_Nu, tr_start = tr_start+cut*(tr_end-tr_start), tr_end = tr_end)
fig_flux.savefig(path_to_report+"/fig_flux.png")
print("Number of neutrino in new 'val' dataset is calculated with its error!")