
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr
from scipy.signal import butter
import os
import glob
import activityTypeAcc

data_folder = "Data\\"  
Data_Name_1 = "sub03_100c.csv"

SR = 50
data1 = pd.read_csv(data_folder+Data_Name_1)[['accX', 'accY', 'accZ']].reset_index(drop=True)
data1 = data1.to_numpy()*1000

begin = 0
count1 = 0
buffer1 = SR
count2 = 0
buffer2 = SR
length = len(data1)
end = length-1
for i in range(1,SR*4):
    if (np.abs(data1[i,0]-data1[i-1,0]) >=10 or np.abs(data1[i,1]-data1[i-1,1]) >=10 or np.abs(data1[i,2]-data1[i-1,2]) >=10) and begin==0:
        count1 = count1 + 1
        buffer1 = SR
    else:
        if buffer1 != 0:
            buffer1 = buffer1 - 1
            count1 = count1 + 1
        else:
            begin = count1

    if (np.abs(data1[length-i,0]-data1[length-i-1,0]) >=10 or np.abs(data1[length-i,1]-data1[length-i-1,1]) >=10 or np.abs(data1[length-i,2]-data1[length-i-1,2]) >=10) and end == length-1:
        count2 = count2 + 1
        buffer2 = SR
    else:
        if buffer2 != 0:
            buffer2 = buffer2 - 1
            count2 = count2 + 1
        else:
            end = length-count2-1

    if i == 4*SR-1:
        if begin == 0:
            begin = i
        if end == length-1:
            end = length-i


data1 = data1[begin:end]
for i in range(3):
    plt.plot(data1[:,i])
    plt.fill_between(np.arange(len(data1[:,i])), 1100, data1[:,i], where= (data1[:,i] > 1100),color='red', alpha=0.4)
    plt.fill_between(np.arange(len(data1[:,i])), data1[:,i], -1100, where= (data1[:,i] < -1100),color='red', alpha=0.4)
plt.axhspan(-1100, 1100, alpha=0.4, color = "green")
plt.ylabel("ACC")
plt.xlabel("Samples")
plt.show()
norm = np.sqrt(np.square(data1[:,0])+np.square(data1[:,1])+np.square(data1[:,2]))

count = 0
buffer = 0
list = []
isbad = 0
for i in range(len(norm)):
    if norm[i] > 1200 or norm[i] <800:
        count = count + 1
        buffer = 20
        if count >= 50 and isbad == 0:
            print("This data is unacceptible")
            isbad = 1
        list.append(1)
    else:
        if buffer != 0:
            buffer = buffer - 1
            count = count + 1
            list.append(1)
        else:
            count = 0   
            list.append(0)

            

fig, (ax1, ax2) = plt.subplots(2, 1)
absdata = abs(data1[:,0])+abs(data1[:,1])+abs(data1[:,2])
ax1.plot(absdata)
ax1.set_ylabel("Sum of Absolute value of ACC in XYZ")
ax1.fill_between(np.arange(len(absdata)), 1100, absdata, where= (absdata > 1100),color='red', alpha=0.4)
ax1.fill_between(np.arange(len(absdata)), absdata, 900, where= (absdata < 900),color='red', alpha=0.4)
ax1.axhspan(900, 1100, alpha=0.4, color = "green")

ax2.set_ylabel("ACC norm")
ax2.set_xlabel("Samples")
ax2.fill_between(np.arange(len(norm)), 1100, norm, where= (norm > 1100),color='red', alpha=0.4)
ax2.fill_between(np.arange(len(norm)), norm, 900, where= (norm < 900),color='red', alpha=0.4)
ax2.plot(norm)
ax2.axhspan(900, 1100, alpha=0.4, color = "green")
plt.show()