import pandas
import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib
import sklearn
import pickle
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from category_encoders import OrdinalEncoder

db = pd.read_csv('Database/Cytotoxicity.csv')
trained_model = joblib.load('ML_models/Trained_model.joblib')
population_size = 50
cell_type = db['Cell type']
test = db['test']
material = db['material']
time = db['time (hr)']
concentration = db['concentration (ug/ml)']
viability = 0  # to be determined by the trained model
hd = db['Hydrodynamic diameter (nm)']
zeta = db['Zeta potential (mV)']
rand_cell_type = random.choice(cell_type)
rand_test = random.choice(test)
rand_material = random.choice(material)
rand_time = random.choice(time)
rand_conc = random.choice(concentration)
rand_hd = random.choice(hd)
rand_zeta = random.choice(zeta)
single = [rand_cell_type, rand_test, rand_material, rand_time, rand_conc, viability, rand_hd, rand_zeta]
#print(single)
single_scaler = [ 5, rand_time, rand_conc, rand_hd, rand_zeta ]
cols = ['del', 'time (hr)', 'concentration (ug/ml)', 'Hydrodynamic diameter (nm)', 'Zeta potential (mV)']
scalar_2d = []
scaler_input = [5, random.choice(time), random.choice(concentration), random.choice(hd), random.choice(zeta)]
for a in range (population_size):
    scalar_2d.append([10, random.choice(time), random.choice(concentration), random.choice(hd), random.choice(zeta) ])
df = pd.DataFrame(scalar_2d , columns= cols)
#print(df)
print('value', type(scalar_2d))
#predict = trained_model.predict(single)
#print(trained_model.summary())
#print(predict)

with open('ML_models/scaler.pkl', 'rb') as f: #working
    scaler_package = pickle.load(f)

samp = scaler_package.fit_transform(df)
#sample_in = scaler.transform(single_scaler)
print(df)
print(samp)
part1 = samp.tolist()[1]
mod_part1 = part1.pop(0)
print(part1)
print(mod_part1)
#x_test = scaler.transform(samp)


#encoder_sample = [rand_cell_type, rand_material, rand_test]
#print(encoder_sample)

with open('ML_models/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

#print(encoder)
encoder_input = [ 0, random.choice(cell_type), random.choice(material), random.choice(test)]
cols = ['unnamed', 'Cell type', 'material', 'test']
inlist = []
for a in range(population_size):
    inlist.append([0, random.choice(cell_type), random.choice(material), random.choice(test)])
df2 = pd.DataFrame(inlist , columns= cols)
#sample = pd.DataFrame(np.array([[0, random.choice(cell_type), random.choice(material), random.choice(test)], columns = cols)
#use_encode = OrdinalEncoder(mapping = encoder, return_df= True )
to_train = encoder.transform(df2)
print(df2.values[1])
print(to_train.values[1])
part2 = to_train.values.tolist()[1]
mod_part2 = part2.pop(0)
print(part1)
print('part 2', part2)
print(mod_part2)
print(mod_part1)
combined_file = part2[1:] + part1

print(combined_file)
#combined = [*mod_part1 , *mod_part2]
#print(combined)
# link https://github.com/scikit-learn-contrib/category_encoders/issues/193
#print(encoder)

#test_y = []
import numpy
print(numpy.array(combined_file).shape)
trained_model = joblib.load('ML_models/Trained_model.joblib')
combined_file = np.reshape(np.array(combined_file), (1, len(combined_file)))
viability_sample = trained_model.predict(combined_file)
print(viability_sample)
