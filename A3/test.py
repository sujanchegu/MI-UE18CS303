import pandas as pd
import numpy as np
from new import *

df = pd.read_csv("./Dataset_Versions/Dataset_V2/train_split.csv")
numpyinput = df[['Weight', 'HB', 'BP']].to_numpy()
numpyoutput = df[['Result_0.0', 'Result_0.0']].to_numpy()
numpyoutput = numpyoutput.transpose()
numpyinput = numpyinput.transpose()

network = NeuralNet()
network.fit(numpyinput, True, 10, numpyoutput)