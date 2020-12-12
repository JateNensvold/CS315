import numpy as np
import pandas as pd
from IPython.display import display

us_yt = pd.read_csv('archive/USvideos.csv')
print(us_yt.columns)

corrolation_list = ['views', 'likes', 'dislikes', 'comment_count']
hm_data = us_yt[corrolation_list].corr() 
display(hm_data)

import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot

matplotlib.pyplot.figure(figsize=(8,6))
sns.heatmap(hm_data, annot=True)

sns.pairplot(us_yt[['views', 'likes']], kind='reg',height=6)