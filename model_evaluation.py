import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

from data_cleaning import clean_data
from feature_engineering import create_features


df = pd.read_csv(
    "../data/cars.csv"
)

df = clean_data(df)

df = create_features(df)

X = df.drop(
    columns=["priceUSD"]
)

y = df["priceUSD"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = joblib.load(
    "../models/car_price_model.joblib"
)

predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)

print("MAE:", mae)

print("RMSE:", rmse)

print("R2:", r2)