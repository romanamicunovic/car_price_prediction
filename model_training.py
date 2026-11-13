import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from data_cleaning import clean_data
from feature_engineering import create_features
from data_preprocessing import create_preprocessor


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

preprocessor = create_preprocessor()

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
    )
])

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "../models/car_price_model.joblib"
)

print(
    "Model uspješno sačuvan."
)