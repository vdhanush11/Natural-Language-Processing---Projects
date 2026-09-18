# Lifecycle stage 3 — Data Understanding
 
import pandas as pd
 
def load_data(path):
    data = pd.read_csv(path, encoding="latin1")
    data = data.rename(columns={"v1": "label", "v2": "message"})
    return data.loc[:, ~data.columns.str.startswith("Unnamed:")]
