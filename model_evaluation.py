import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib
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
rand_cell_type = random.choice(cell_type)
rand_test = random.choice(test)
rand_material = random.choice(material)
rand_time = random.choice(time)
rand_conc = random.choice(concentration)
rand_hd = random.choice(hd)
rand_zeta = random.choice(zeta)
single = [rand_cell_type, rand_test, rand_material, rand_time, rand_conc, viability, rand_hd, rand_zeta]
print(single)

predict = trained_model.predict(single)
print(trained_model.summary())
print(predict)
