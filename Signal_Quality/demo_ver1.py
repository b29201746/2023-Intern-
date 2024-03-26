import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr
from scipy.signal import butter
import os
import glob
import pingouin as pg

def correlation(a, b):
    cor, _ = pearsonr(a, b)
    return cor
def autocorrelation(data1, data2, window, maxshift):
    lcount = 0
    rcount = 0
    size = min(len(data1), len(data2))
    for n in range(0, size, window):
        if size-max(lcount, rcount) <= n+window+maxshift:
            data1 = data1[:n-window]
            data2 = data2[:n-window]
            break
        best = 0
        shift = 0
        dir = 0
        if n != 0:
            maxshift = 10
        for l in range(maxshift):
            a = correlation(data1[n+l:n+window+l],data2[n:n+window])
            b = correlation(data1[n:n+window],data2[n+l:n+window+l])
            
            if best<a:
                best = a
                shift = l
                dir = 0
            if best<b:
                best = b
                shift = l
                dir = 1
        if dir == 0:
            data1 = np.concatenate((data1[:n],data1[n+shift:n+window+shift],data1[n+window+shift:]), axis=0)
            lcount = lcount + shift
        else:
            data2 = np.concatenate((data2[:n],data2[n+shift:n+window+shift],data2[n+window+shift:]), axis=0)
            rcount = rcount + shift
        # print(shift)
    return data1, data2

Ref_data_path = input("Enter the filepath of the reference data : ")
data2 = pd.read_csv(Ref_data_path)[['accX', 'accY', 'accZ']][500:-500].reset_index(drop=True)
Test_data_path = input("Enter the filepath of the test data : ")
data1 = pd.read_csv(Test_data_path)[['accX', 'accY', 'accZ']][500:-500].reset_index(drop=True)

data1 = data1.to_numpy()
data2 = data2.to_numpy()

axis = ["x", "y", "z"]
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_title('Plots of XYZ')
p = [ax1, ax2, ax3]
for i in range(3):
    a = data1[:,i]
    b = data2[:,i]
    p[i].plot(a, label = "Test_Data_"+axis[i])
    p[i].plot(b, label = "Ref_Data_"+axis[i])
    p[i].legend()
plt.show()


data1 = np.sqrt(np.square(data1[:,0])+np.square(data1[:,1])+np.square(data1[:,2]))
data2 = np.sqrt(np.square(data2[:,0])+np.square(data2[:,1])+np.square(data2[:,2]))

b, a = butter(N=2, Wn=[0.8,1.15],btype='bandpass', fs=50)
data1 = scipy.signal.filtfilt(b, a, data1)
data2 = scipy.signal.filtfilt(b, a, data2)

data1, data2 = autocorrelation(data1, data2, 100, 50)

plt.plot(data1[:700], label="Test_Data")
plt.plot(data2[:700], label="Ref_Data")
plt.title("Plot of ACC_Norm after preprocess")
plt.xlabel("Sample")
plt.ylabel("ACC_Norm")
plt.axhspan(0.6,1.5, facecolor='green', alpha=0.3)

plt.legend()
plt.show()

x_mean = np.mean(data1)
y_mean = np.mean(data2)
a1 = np.sum(np.multiply((data1-x_mean), (data2-y_mean)))
a1 = a1/(np.sum(np.square(data2-y_mean)))
a0 = x_mean - a1*y_mean
# r = np.sum(np.square(a1*data2+a0-x_mean))/np.sum(np.square(data1-x_mean))
tmp = a1*data2+a0

plt.scatter(data2, data1, s=8)
plt.title("Plot of Reference Data versus Test Data")
plt.ylabel("Test_Data")
plt.xlabel("Ref_Data")
plt.plot(data2, tmp)
plt.show()

data1 = pd.DataFrame(data1, columns=['norm'])
data2 = pd.DataFrame(data2, columns=['norm'])
# plt.plot(data1[col], label="data1")
# plt.plot(data2[col], label="data2")
# plt.legend()
# plt.show()
col ='norm'

data1 = data1[col].reset_index(drop=True)
data2 = data2[col].reset_index(drop=True)
rater1 = pd.DataFrame('1', index=np.arange(len(data1)), columns=['rater'])
rater2 = pd.DataFrame('2', index=np.arange(len(data2)), columns=['rater'])
target1 = pd.DataFrame(np.arange(len(data1)), index=np.arange(len(data1)), columns=['target'])
target2 = pd.DataFrame(np.arange(len(data2)), index=np.arange(len(data2)), columns=['target'])

result = pd.concat([data1,data2], axis=0)
rater = pd.concat([rater1, rater2], axis=0)
target = pd.concat([target1, target2], axis=0)
result = pd.concat([result, rater, target], axis = 1)
icc = pg.intraclass_corr(data=result, ratings=col, raters='rater', targets= 'target')

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.axhline(y=correlation(data1, data2)**2, label="R2")
ax1.margins(0)
ax1.grid(linewidth=0.5, axis="y", c="black",linestyle="--")
ax1.axhspan(0.5,1, facecolor='green', alpha=0.3)
ax1.axhspan(0,0.5, facecolor='red', alpha=0.3)
ax1.set_xticks([])
ax1.set_yticks(np.arange(0, 1, 0.1))
ax1.set_title("R2")
ax1.legend()

ax2.axhline(icc['ICC'][5], label="ICC")
ax2.margins(0)
ax2.grid(linewidth=0.5, axis="y", c="black",linestyle="--")
ax2.axhspan(0.65,1, facecolor='green', alpha=0.3)
ax2.axhspan(0,0.65, facecolor='red', alpha=0.3)
ax2.set_xticks([])
ax2.set_yticks(np.arange(0, 1, 0.1))
ax2.set_title("ICC")
ax2.legend()

plt.show()