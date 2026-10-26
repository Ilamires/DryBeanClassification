import os
import pandas as pd

def make_csv(xlsx_path, csv_path, arff_path = "") -> pd.DataFrame:
    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        from scipy.io.arff import loadarff
        data, meta = loadarff(arff_path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if isinstance(df[col].iloc[0], bytes):
                df[col] = df[col].str.decode("utf-8")
    df.to_csv(csv_path, index=False)
    print(f"CSV создан: {csv_path}")
    return df


def load_data(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)