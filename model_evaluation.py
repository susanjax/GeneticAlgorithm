import pandas
import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib
import sklearn
import pickle
import numpy as np
from category_encoders import OrdinalEncoder
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
rand_cell_type = random.choice(cell_type)
rand_test = random.choice(test)
rand_material = random.choice(material)
rand_time = random.choice(time)
rand_conc = random.choice(concentration)
rand_hd = random.choice(hd)
rand_zeta = random.choice(zeta)
single = [rand_cell_type, rand_test, rand_material, rand_time, rand_conc, viability, rand_hd, rand_zeta]
print(single)

#predict = trained_model.predict(single)
#print(trained_model.summary())
#print(predict)

with open('ML_models/scaler.pkl', 'rb') as f: #working
    scaler = pickle.load(f)
print(scaler)
#value_scaler = scaler.fit_transform(values)

#to_transform = [rand_time, rand_conc, rand_hd, rand_zeta]
#print(to_transform)

#to_trasform_encoder = [rand_cell_type, rand_material, rand_test]
#print(to_trasform_encoder)

with open('ML_models/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

#print(encoder)
sample = pd.DataFrame(np.array([['PC12', 'Ag', 'MTT'], ['SW480', 'Au', 'LDH']]), columns = ['Cell type', 'material', 'test'])
#use_encode = OrdinalEncoder(mapping = encoder, return_df= True )
#to_train = encoder.fit_transform(sample)
#print(to_train)
# link https://github.com/scikit-learn-contrib/category_encoders/issues/193
print(encoder)