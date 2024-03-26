import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr
from scipy.signal import butter
import os
import glob
import pingouin as pg
import activityTypeAcc
import warnings


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


def acitivtyclassifier(data1):
    activity1 = activityTypeAcc.ActivityTypeAcc()
    acctype1 = []

    count_act = 0
    count_stc = 0
    start1 = 0
    end1 = 0

    for i in range(0, len(data1),50):
        t1, cad = activity1.update(accX=data1[i:i+50,0], accY=data1[i:i+50,1], accZ=data1[i:i+50,2])
        if(t1 == 1):
            count_act = count_act + 1
            if count_act == 5:
                start1 = i -4*50
            if i+50 >= len(data1) and end1 == 0:
                end1 = i
            count_stc = 0
        else:
            if start1 != 0:
                count_stc = count_stc + 1
                if count_stc == 6:
                    end1 = i - 5*50
                    acctype1.append(t1)
        if i+50 >= len(data1) and end1 == 0 and start1!=0:
            end1 = i - count_stc*50
        acctype1.append(t1)
        
    data1 = data1[start1:end1]
        
    if end1-start1 < 3000:
        print("The data is invalid")
        print("Its walking time is less than 60 seconds")
        quit()
    
    return data1

warnings.filterwarnings('ignore')
Ref_data_path = 'Criteria\\'

file_pattern = os.path.join(Ref_data_path, "*c.csv")
match_files = glob.glob(file_pattern)


num = 0

all_data = []
for file_path in match_files:
    data = acitivtyclassifier(pd.read_csv(file_path).to_numpy()*1000)
    data = np.sqrt(np.square(data[:,0])+np.square(data[:,1])+np.square(data[:,2]))
    all_data.append(data)

while len(all_data)!=1:
    next_data = []
    i = 0
    if(len(all_data)%2 != 0):
        next_data.append(all_data[0])
        i = 1
    while i < len(all_data):
        d1, d2 = autocorrelation(all_data[i], all_data[i+1], 100, 50)
        d1 = np.add(d1, d2)/2
        next_data.append(d1)
        i = i+2
    all_data = next_data
    

plt.plot(all_data[0])
plt.show()
all_data = pd.DataFrame(all_data[0], columns=["norm"])
all_data.to_csv(Ref_data_path+"Mean.csv", index=None)