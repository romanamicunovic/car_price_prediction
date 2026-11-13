import pandas as pd


def clean_data(df):

    # Uklanjanje duplikata
    df = df.drop_duplicates()

    # Standardizacija naziva kolona
    df.columns = df.columns.str.strip()

    return df