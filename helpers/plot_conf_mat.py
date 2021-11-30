import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# array = np.array([[1474/1562, 84/1562, 4/1562],
#          [71/3127, 3045/3127, 11/3127],
#          [11/1560, 271/1560, 1278/1560]]).T

# array = np.array([[1474, 84, 4],
#          [71, 3045, 11],
#          [11, 271, 1278]]).T

# array = np.array([[122, 40, 17],
#          [46, 290, 57],
#          [5, 16, 99]])

array = np.array([[122/173, 40/346, 17/173],
         [46/173, 290/346, 57/173],
         [5/173, 16/346, 99/173]])

df_cm = pd.DataFrame(array, ["Positive", "Neutral", "Negative"], ["Positive", "Neutral", "Negative"])
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Reds')
# sn.heatmap(df_cm, fmt='', annot=True, annot_kws={"size": 16}, cmap='Reds') # font size

plt.show()
plt.savefig("./confusion_matrix_test.png")