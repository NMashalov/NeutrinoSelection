from my_analysis import *
from IPython.display import clear_output

# data and model's names
data_names = [n for n in os.listdir('../data/') if n.endswith('.h5')]
for i,h5n in enumerate(data_names):
    print(str(i+1),". "+h5n)
i = int(input("Which dataset do you want to use? Print it's number! \n"))
name = data_names[i-1]
path_to_h5 = '../data/' + name

model_names = [n for n in os.listdir('../trained_models/') if not n.startswith('logs') and not n.startswith('.')]
for i,mn in enumerate(model_names):
    print(str(i+1),". "+mn)
i = int(input("Which model do you want to choose? Print it's number! \n"))
model_name = model_names[i-1]

for i,r in enumerate(['train','test','val']):
    print(str(i+1),r)
i = int(input("Which regime do you want to choose? Print it's number! \n"))
regime = ['train','test','val'][i-1]

path_to_model = '../trained_models/' + model_name +'/'+'best' #Change to best later!
model = tf.keras.models.load_model(path_to_model, compile=False)
model._name = model_name
clear_output(wait=False)
_ = make_preds(model, regime, path_to_h5)
print(f"{model_name} predictions for {regime} are made!")