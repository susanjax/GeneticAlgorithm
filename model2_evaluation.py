import pandas
from category_encoders import OrdinalEncoder

ordinal_cols_mapping = [{
    "col": "GarageCond",
    "mapping": {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': 0
    }}, {
    "col": "GarageQual",
    "mapping": {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': 0
    }}
]

train = pandas.DataFrame(data=[['Ex', 'Po'], ['Po', 'Ex'], ['TA', 'Ex']], columns=['GarageCond', 'GarageQual'])
encoder = OrdinalEncoder(mapping = ordinal_cols_mapping, return_df = True)
df_train = encoder.fit_transform(train)
print(df_train)