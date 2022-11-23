
import librosa 
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def extract_features(audio):
  aud, rate = librosa.load(audio, res_type="kaiser_best")
  mfcc_feature = librosa.feature.mfcc(y=aud, sr=rate, n_mfcc=100)
  return np.mean(mfcc_feature.T, axis=0)

def extract(path1, path2):
  extracted_features = []
  for i in os.listdir(path1):
    extracted_features.append([extract_features(os.path.join(path1, i)), 1])
  for j in os.listdir(path2):
    extracted_features.append([extract_features(os.path.join(path2, j)), 0])
  
  df = pd.DataFrame(extracted_features, columns=["data", "target"])
  df1 = df.sample(len(df))
  df1.to_csv("result.csv", index=False)
  return df1





