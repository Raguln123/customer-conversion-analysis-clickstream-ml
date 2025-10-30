import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    df = df.fillna(df.mean(numeric_only=True))
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    scaler = StandardScaler()
    df[df.select_dtypes(include='number').columns] = scaler.fit_transform(
        df[df.select_dtypes(include='number').columns])
    return df
