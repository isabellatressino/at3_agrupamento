### arquivo com algumas funcoes que vamos utilizar na versao final

def normalize_df(df: pd.DataFrame):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    
def transform_in_df(scaled_df: np.ndarray, data: tuple[Bunch, tuple]):
    df = pd.DataFrame(scaled_df, columns= data.feature_names)