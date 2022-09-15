# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
#
# #Import clean property data
# real_estate = pd.read_csv(r'/Users/marianafarr/PycharmProjects/CapstoneFinal/real-estate-final.csv')
# # zip code with leading zeros
# real_estate['zip_code'] = real_estate['zip_code'].astype(str).str.zfill(5)
# #Drop any columns with NAN
# real_estate = real_estate.dropna(axis=1)
# # Get a DF of just unique zip codes
# zip_codes = real_estate['zip_code'].unique()
#
#
#
# # Split into training and test data
#
# np.random.seed(42)
# # Create Data, make price the target column, and drop irrelevant columns
# y = real_estate["price"]
# X = real_estate.drop(['price', 'state', 'acres_in_feet', 'total_size_ft'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = RandomForestRegressor()
# model.fit(X_train, y_train)
# model.score(X_test, y_test)
#
# #pickle the model
# joblib.dump(model, "model.pkl")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Import clean property data
real_estate = pd.read_csv(r'real-estate-final.csv')
# zip code with leading zeros
real_estate['zip_code'] = real_estate['zip_code'].astype(str).str.zfill(5)
#Drop any columns with NAN
real_estate = real_estate.dropna(axis=1)
# Get a DF of just unique zip codes
zip_codes = real_estate['zip_code'].unique()


# Split into training and test data

np.random.seed(42)
# Create Data, make price the target column, and drop irrelevant columns
y = real_estate["price"]
X = real_estate.drop(['price', 'state', 'acres_in_feet', 'total_size_ft'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)


#pickle the model
joblib.dump(model, "model.pkl")
