import pandas as pd
import random
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from category_encoders import OrdinalEncoder
import matplotlib.pyplot as plt
import sklearn

db = pd.read_csv('Database/Cytotoxicity.csv')
trained_model = joblib.load('ML_models/Trained_model.joblib')

cell_type = db['Cell type']
test = db['test']
material = db['material']
time = db['time (hr)']
concentration = db['concentration (ug/ml)']
viability = 0  # to be determined by the trained model
hd = db['Hydrodynamic diameter (nm)']
zeta = db['Zeta potential (mV)']


encoder_input = [ 0, random.choice(cell_type), random.choice(material), random.choice(test)]
cols = ['unnamed', 'Cell type', 'material', 'test']
inlist = []
for a in range(10):
    inlist.append([0, random.choice(cell_type), random.choice(material), random.choice(test)])
df = pd.DataFrame(inlist , columns= cols)
with open('ML_models/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
out1 = encoder.transform(df) #randomized sample with 3 features (one unnecessary)
mod_out1 = out1.iloc[: , 1:]
#print(type(mod_out1))
#print('part_1:', df.values.tolist()[0][1:], mod_out1)

scaler_input = []
scaler_in = [0, random.choice(time), random.choice(concentration), random.choice(hd), random.choice(zeta) ]
cols = ['del', 'time (hr)', 'concentration (ug/ml)', 'Hydrodynamic diameter (nm)', 'Zeta potential (mV)']
for a in range (10):
    scaler_input.append([10, random.choice(time), random.choice(concentration), random.choice(hd), random.choice(zeta) ])
df2 = pd.DataFrame(scaler_input, columns= cols)
with open('ML_models/scaler.pkl', 'rb') as f:
    sp = pickle.load(f)
out2 = sp.fit_transform(df2)
#print(type(out2))
mod_out2 = np.delete(out2, 0, axis = 1)
mod_out2 = pd.DataFrame(mod_out2, columns = ['a', 'b', 'c', 'd'])
#print(mod_out2)
#print('part_2:', df2.values.tolist()[0][1:], mod_out2 )

sample = pd.concat([mod_out1, mod_out2], axis=1, join= "inner")
normal_cell = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for a in range(10):
    sample1 = sample.iloc[a].values.tolist()
    if normal_cell.count(sample1[0]) == True:
        cell_viability = trained_model.predict([sample1])
        cancer_cell = sample1
        cancer_cell[0] = 1 # here put a number which is cancer cell
        cancer_viability = trained_model.predict([cancer_cell])
        fitness = normal_cell/(normal_cell + cancer_cell)
        break
    else:
        a+= 1
        continue
    break
print(sample1)


#sample_reshape = np.reshape(np.array(sample), (1, len(sample1)))
print(type(sample1))
print(normal_cell.count(4))
print(sample1[0])
a = 0
#if sample1[a][2] == normal_cell.count(sample1[0][a]):
#    normal = samp
#viability = trained_model.predict([sample1])
#print(viability)

