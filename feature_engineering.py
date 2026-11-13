from datetime import datetime


def create_features(df):

    current_year = datetime.now().year

    df["car_age"] = (
        current_year -
        df["year"]
    )

    df["engine_volume_liters"] = (
        df["volume(cm3)"] / 1000
    )

    df["mileage_per_year"] = (
        df["mileage(kilometers)"] /
        (df["car_age"] + 1)
    )

    return df