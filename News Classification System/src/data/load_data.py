# Lifecycle stage 3 — Data Understanding
 
import glob
import pandas as pd
 
 
def load_data(folder):
 
    files = glob.glob(f"{folder}/inshort_news_data-*.csv")
 
    frames = [pd.read_csv(f) for f in files]
 
    return pd.concat(frames, ignore_index=True)
