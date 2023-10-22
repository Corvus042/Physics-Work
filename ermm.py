import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("C:/Users/Nwgol/Onedrive/Desktop/uv-vis-spectra-thickness-series.csv")
W1 = data['one-eighth W'].tolist()
T1 = data['one-eighth T'].tolist()
W1 = (W1[1:403])
T1 = (T1[1:403])

plt.plot(W1,T1)
plt.show()