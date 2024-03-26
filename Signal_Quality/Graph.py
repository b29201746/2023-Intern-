import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = "ICC_outcome_3"
input = pd.read_csv(file_name+".csv", index_col=None)

for i in range(1,6):
    plt.scatter(np.arange(1,len(input)+1), input.iloc[:,i], s = 15)

plt.margins(0)
plt.axvspan(0,5.5, facecolor='red', alpha=0.2)
plt.axvspan(5.5,10.5, facecolor='yellow', alpha=0.2)
plt.axvspan(10.5,13.5, facecolor='blue', alpha=0.2)
plt.axvspan(13.5,15.5, facecolor='green', alpha=0.2)
plt.axvspan(15.5,16.5, facecolor='white', alpha=0.2)
plt.grid(linewidth=0.5)
plt.xticks(range(0, len(input) + 1, 1))
plt.yticks(np.arange(0, 1, 0.1))
plt.title(file_name)
plt.xlabel("task")
if file_name[0]=='R':
    plt.ylabel("R2")
else:
    plt.ylabel("ICC")
plt.show()
