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

def correlation(a, b):
    cor, _ = pearsonr(a, b)
    #print(str(cor**2))
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
    return data1, data2  


data_folder = "Data\\"
Data_Name_1 = ["sub01_100c.csv", "sub02_100c.csv", "sub03_100c.csv", "sub04_100c.csv", "sub05_100c.csv"]

folder_path = "Data\\" 
file_pattern = os.path.join(folder_path, "*.csv")
match_files = glob.glob(file_pattern)
output_file = pd.DataFrame()

for k in range(5):
    wfile = pd.DataFrame([], columns=[Data_Name_1[k]])
    for file_path in match_files:
        data1 = pd.read_csv(data_folder+Data_Name_1[k])[['accX', 'accY', 'accZ']][:].reset_index(drop=True)
        data2 = pd.read_csv(file_path)[['accX', 'accY', 'accZ']][:].reset_index(drop=True)
        # plt.plot(data1['accX'], label="1")
        # plt.plot(data2['accX'], label="2")
        # plt.legend()
        # plt.show()

        data1 = data1.to_numpy()*1000
        data2 = data2.to_numpy()*1000

        activity1 = activityTypeAcc.ActivityTypeAcc()
        activity2 = activityTypeAcc.ActivityTypeAcc()
        acctype1 = []
        acctype2 = []

        count_act = 0
        count_stc = 0
        start1 = 0
        start2 = 0
        end1 = 0
        end2 = 0

        for i in range(0, len(data1),50):
            t1, cad = activity1.update(accX=data1[i:i+50,0], accY=data1[i:i+50,1], accZ=data1[i:i+50,2])
            if(t1 == 1):
                count_act = count_act + 1
                if count_act == 5:
                    start1 = i -4*50
                if count_stc < 5:
                    count_stc = 0
            else:
                if start1 != 0:
                    count_stc = count_stc + 1
                    if count_stc == 6:
                        end1 = i - 5*50
                        acctype1.append(t1)
            if i+50 >= len(data1) and end1 == 0 and start1 != 0:
                end1 = i
            acctype1.append(t1)

        count_act = 0
        count_stc = 0
        for i in range(0, len(data2),50):
            t2, cad = activity2.update(accX=data2[i:i+50,0], accY=data2[i:i+50,1], accZ=data2[i:i+50,2])
            if(t2 == 1):
                count_act = count_act + 1
                if count_act == 5:
                    start2 = i-4*50
                if count_stc < 5:
                    count_stc = 0
            else:
                if start2 != 0:
                    count_stc = count_stc + 1
                    if count_stc == 6:
                        end2 = i - 5*50
                        acctype2.append(t2)
            if i+50 >= len(data2) and end2 == 0 and start2 != 0:
                end2 = i
            acctype2.append(t2)
        
        data1 = data1[start1:end1]
        data2 = data2[start2:end2]
        
        print(end2-start2)
        if(end1-start1 < 3000 or end2-start2 < 3000):
            print(Data_Name_1[k],start1, end1)
            print(file_path,start2, end2)
            print("The Data is not walking")
            wdata = pd.DataFrame([0], index=[file_path[5:]],columns=[Data_Name_1[k]])
            wfile = pd.concat([wfile, wdata], axis=0)
            continue

        data1 = np.sqrt(np.square(data1[:,0])+np.square(data1[:,1])+np.square(data1[:,2]))
        data2 = np.sqrt(np.square(data2[:,0])+np.square(data2[:,1])+np.square(data2[:,2]))

        b, a = butter(N=2, Wn=[0.8,1.15],btype='bandpass', fs=50)
        data1 = scipy.signal.filtfilt(b, a, data1)
        data2 = scipy.signal.filtfilt(b, a, data2)

        col = "norm"
        data1, data2 = autocorrelation(data1, data2, 100, 50)

        data1 = pd.DataFrame(data1, columns=['norm'])
        data2 = pd.DataFrame(data2, columns=['norm'])
        # plt.plot(data1[col], label="data1")
        # plt.plot(data2[col], label="data2")
        # plt.legend()
        # plt.show()


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
        wdata = pd.DataFrame([icc['ICC'][5]], index=[file_path[5:]],columns=[Data_Name_1[k]])
        wfile = pd.concat([wfile, wdata], axis=0)
        print(file_path + " = " + str(icc['ICC'][5]))
    if k == 0:
        output_file = wfile
    else:
        output_file = pd.concat([output_file,wfile], axis=1)

output_file.to_csv("C:\\Users\\b2920\\OneDrive\\Desktop\\test\\ICC_outcome_3.csv")


# axis = ["x", "y", "z"]
# fig, (ax1, ax2, ax3) = plt.subplots(3,1)
# a = [ax1, ax2, ax3]

# for i in range(3):

#     a[i].plot(x, label="maxim_"+axis[i])
#     a[i].plot(y, label="naxsen_"+axis[i])
#     a[i].legend()
#     x = pd.DataFrame(x, columns=['maxim'])
#     y = pd.DataFrame(y, columns=['naxsen'])
#     rater = pd.DataFrame('A', index=np.arange(len(x)), columns=['rater'])
#     result = pd.concat([x,rater,y], axis=1)
#     icc = pg.intraclass_corr(data=result, targets='naxsen', raters='rater', ratings='maxim')
#     print(icc)


# plt.plot(data1[col], label="max")
# plt.plot(data2[col], label="nax")
# plt.legend()
# plt.show()

# mean_nax = np.mean(data2[col])
# mean_max = np.mean(data1[col])
# a1 = np.sum(np.multiply((data1[col]-mean_max), (data2[col]-mean_nax)))
# a1 = a1/(np.sum(np.square(data2[col]-mean_nax)))

# a0 = mean_max - a1*mean_nax


# r2 = np.sum(np.square(a0+a1*data2[col]-mean_max))/np.sum(np.square(data1[col]-mean_max))
# print(r2)

# r22 = 1 - np.sum(np.square(a0+a1*data2[col]-data1[col]))/np.sum(np.square(data1[col]-mean_max))
# print(r22)