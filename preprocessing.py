from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize_data(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

