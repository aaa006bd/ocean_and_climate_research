import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()

df = pd.read_csv('output-init-287.15.csv')

sns.lineplot(x='time', y='output', data=df)

plt.show()