
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1_name = "sub01_100c.csv"

data1 = pd.read_csv("Data\\"+data1_name)[['accX', 'accY', 'accZ']]
# data3 = pd.read_csv("Data\\random100s_2.csv")[['accX', 'accY', 'accZ']]
# data4 = pd.read_csv("Data\\random100s_3.csv")[['accX', 'accY', 'accZ']]
# data5 = pd.read_csv("Data\\stationary100s.csv")[['accX', 'accY', 'accZ']]



data2 = data1.to_numpy()
# data3 = data3.to_numpy()
# data4 = data4.to_numpy()
# data5 = data5.to_numpy()


data2 = np.sqrt(np.square(data2[:,0])+np.square(data2[:,1])+np.square(data2[:,2]))
# data3 = np.sqrt(np.square(data3[:,0])+np.square(data3[:,1])+np.square(data3[:,2]))
# data4 = np.sqrt(np.square(data4[:,0])+np.square(data4[:,1])+np.square(data4[:,2]))
# data5 = np.sqrt(np.square(data5[:,0])+np.square(data5[:,1])+np.square(data5[:,2]))

plt.plot(data2, label='sub01')
# plt.plot(data3, label='random2')
# plt.plot(data4, label='random3')
# plt.plot(data5, label='stationary')
plt.legend()
plt.show()
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax2.plot(maxim["ACC_X"], label="ACC_X")
# ax2.plot(maxim["ACC_Y"], label="ACC_Y")
# ax2.plot(maxim["ACC_Z"], label="ACC_Z")

# ax1.plot(naxsen["accX"], label="ACC_X")
# ax1.plot(naxsen["accY"], label="ACC_Y")
# ax1.plot(naxsen["accZ"], label="ACC_Z")


# ax1.grid(), ax1.legend()
# ax2.grid(), ax2.legend()
# plt.show()

