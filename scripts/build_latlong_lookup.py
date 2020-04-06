import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../shape raw/geocodes.csv')
plt.plot(df['longitude'], df['latitude'], 'bo')
plt.show()
