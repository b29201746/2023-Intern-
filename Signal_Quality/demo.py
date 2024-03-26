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
    flag = 0
    size = min(len(data1), len(data2))
    for n in range(0, size, window):
        if size-max(lcount, rcount) <= n+window+maxshift:
            data1 = data1[:n-window]
            data2 = data2[:n-window]
            break
        best = 0
        shift = 0
        dir = 0
        if n != 0 and flag == 0:
            maxshift = maxshift//5
            flag = 1
            
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


def acitivtyclassifier(data1, SR):
    activity1 = activityTypeAcc.ActivityTypeAcc()
    acctype1 = []

    count_act = 0
    count_stc = 0
    start1 = 0
    end1 = 0

    for i in range(0, len(data1),SR):
        t1, cad = activity1.update(accX=data1[i:i+SR,0], accY=data1[i:i+SR,1], accZ=data1[i:i+SR,2])
        if(t1 == 1):
            count_act = count_act + 1
            if count_act == 5:
                start1 = i -4*SR
            if i+SR >= len(data1) and end1 == 0:
                end1 = i
            count_stc = 0
        else:
            if start1 != 0:
                count_stc = count_stc + 1
                if count_stc == 6:
                    end1 = i - 5*SR
                    acctype1.append(t1)
        if i+SR >= len(data1) and end1 == 0 and start1!=0:
            end1 = i - count_stc*SR
        acctype1.append(t1)
        
    data1 = data1[start1:end1]
        
    if end1-start1 < SR*60:
        print("The data is invalid")
        print("Its walking time is less than 60 seconds")
        quit()
    
    return data1

warnings.filterwarnings('ignore')
Ref_data_path = 'Criteria\\'

Test_data_path = input("Enter the filepath of the test data : ")
Test_data_unit = input("Unit (mg / g):")
Test_data_unit = 1000 if Test_data_unit == "g" else 1
Test_data_SR = input("Sample Rate:")
Test_data_SR = int(Test_data_SR)
test_data = pd.read_csv(Test_data_path)[['accX', 'accY', 'accZ']][:-300].reset_index(drop=True)

Ref_data_SR = 50
Ref_data_unit = 1000

General_SR = 0
Type = input("Enter the type of the test data (static/active):")

if Type == "static":
    data1 = pd.read_csv(Test_data_path)[['accX', 'accY', 'accZ']].reset_index(drop=True)
    data1 = data1.to_numpy()*Test_data_unit

    begin = 0
    count1 = 0
    buffer1 = Test_data_SR*2
    count2 = 0
    buffer2 = Test_data_SR*2
    length = len(data1)
    end = length-1
    for i in range(1,Test_data_SR*5): # 去除開頭跟結尾的雜訊
        if (np.abs(data1[i,0]-data1[i-1,0]) >=10 or np.abs(data1[i,1]-data1[i-1,1]) >=10 or np.abs(data1[i,2]-data1[i-1,2]) >=10) and begin==0:
            count1 = count1 + 1
            buffer1 = Test_data_SR
        else:
            if buffer1 != 0:
                buffer1 = buffer1 - 1
                count1 = count1 + 1
            else:
                begin = count1

        if (np.abs(data1[length-i,0]-data1[length-i-1,0]) >=10 or np.abs(data1[length-i,1]-data1[length-i-1,1]) >=10 or np.abs(data1[length-i,2]-data1[length-i-1,2]) >=10) and end == length-1:
            count2 = count2 + 1
            buffer2 = Test_data_SR
        else:
            if buffer2 != 0:
                buffer2 = buffer2 - 1
                count2 = count2 + 1
            else:
                end = length-count2-1

        if i == 4*Test_data_SR-1:
            if begin == 0:
                begin = i
            if end == length-1:
                end = length-i


    data1 = data1[begin:end]
    axis = ['x', 'y', 'z']
    for i in range(3):
        plt.plot(data1[:,i], label=axis[i])
        plt.fill_between(np.arange(len(data1[:,i])), 1100, data1[:,i], where= (data1[:,i] > 1100),color='red', alpha=0.4, label= "unaccpetible" if i == 2 else None)
        plt.fill_between(np.arange(len(data1[:,i])), data1[:,i], -1100, where= (data1[:,i] < -1100),color='red', alpha=0.4)
    plt.axhspan(-1100, 1100, alpha=0.4, color = "green", label="acceptible")

    plt.legend()
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
            buffer = Test_data_SR
            if count >= Test_data_SR*2 and isbad == 0:
                print("This data is invalid")
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

    if isbad == 0:
        print("This data is valid")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    absdata = abs(data1[:,0])+abs(data1[:,1])+abs(data1[:,2])
    ax1.plot(absdata)
    ax1.fill_between(np.arange(len(absdata)), 1100, absdata, where= (absdata > 1100),color='red', alpha=0.5, label="unacceptible")
    ax1.fill_between(np.arange(len(absdata)), absdata, 900, where= (absdata < 900),color='red', alpha=0.5)
    ax1.axhspan(900, 1100, alpha=0.5, color = "green", label="acceptible")
    ax1.legend()
    ax1.set_ylabel("Sum of Absolute ACC in XYZ")
    ax2.fill_between(np.arange(len(norm)), 1100, norm, where= (norm > 1100),color='red', alpha=0.5, label="unacceptible")
    ax2.fill_between(np.arange(len(norm)), norm, 900, where= (norm < 900),color='red', alpha=0.5)
    ax2.plot(norm)
    ax2.axhspan(900, 1100, alpha=0.5, color = "green", label="acceptible")
    ax2.set_ylabel("ACC norm")
    ax2.set_xlabel("Samples")
    ax2.legend()
    plt.show()
if Type == "active":
    file_pattern = os.path.join(Ref_data_path, "*c.csv")
    match_files = glob.glob(file_pattern)

    test_data = test_data.to_numpy()*Test_data_unit
    
    if Test_data_SR > Ref_data_SR:
        test_data = test_data[::Test_data_SR//Ref_data_SR]
        Test_data_SR = Ref_data_SR
    
    test_data = acitivtyclassifier(test_data, Test_data_SR)

    axis = ["x", "y", "z"]
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.set_title('Plots of XYZ')
    p = [ax1, ax2, ax3]
    for i in range(3):
        a = test_data[:,i]
        p[i].plot(a, label = "Test_Data_"+axis[i])
        p[i].legend()
    fig.supxlabel('Samples')
    fig.supylabel('ACC (mg)')
    plt.show()

    icc_result = pd.DataFrame([],columns=['ICC'])
    r2_result = pd.DataFrame([],columns=['R2'])
    data1 = np.sqrt(np.square(test_data[:,0])+np.square(test_data[:,1])+np.square(test_data[:,2]))
    num = 0

    resample = 1
    refmean = pd.read_csv(Ref_data_path+"Mean.csv")
    if Ref_data_SR > Test_data_SR:
        resample = Ref_data_SR//Test_data_SR
        refmean = refmean[::resample].to_numpy()
    else:
        refmean = refmean.to_numpy()
    refmean = refmean.reshape([len(refmean)])

    showdata1, showdata2 = autocorrelation(refmean, data1, 2*Test_data_SR, Test_data_SR)
    plt.title("Plot of segment of ACC norm after autocorrelation")
    plt.plot(showdata1[:500], label="Mean of Reference Data")
    plt.plot(showdata2[:500], label="Test Data")
    plt.xlabel("Sample")
    plt.ylabel("ACC norm (mg)")
    plt.legend()    
    plt.show()

    print("Please wait for seconds...")
    for file_path in match_files:
        print("Comparing with the reference data "+str(num))
        data2 = pd.read_csv(file_path)[['accX', 'accY', 'accZ']][::resample].to_numpy()*Ref_data_unit
        data2 = acitivtyclassifier(data2, Test_data_SR)
        data2 = np.sqrt(np.square(data2[:,0])+np.square(data2[:,1])+np.square(data2[:,2]))
        num = num+1

        b, a = butter(N=2, Wn=[0.8,1.15],btype='bandpass', fs=Test_data_SR)
        smoothdata1 = scipy.signal.filtfilt(b, a, data1)
        smoothdata2 = scipy.signal.filtfilt(b, a, data2)
        smoothdata1, smoothdata2 = autocorrelation(smoothdata1, smoothdata2, 2*Test_data_SR, Test_data_SR)

        smoothdata1 = pd.DataFrame(smoothdata1, columns=['norm']).reset_index(drop=True)
        smoothdata2 = pd.DataFrame(smoothdata2, columns=['norm']).reset_index(drop=True)

        col ='norm'

        smoothdata1 = smoothdata1[col].reset_index(drop=True)
        smoothdata2 = smoothdata2[col].reset_index(drop=True)
        rater1 = pd.DataFrame('1', index=np.arange(len(smoothdata1)), columns=['rater'])
        rater2 = pd.DataFrame('2', index=np.arange(len(smoothdata2)), columns=['rater'])
        target1 = pd.DataFrame(np.arange(len(smoothdata1)), index=np.arange(len(smoothdata1)), columns=['target'])
        target2 = pd.DataFrame(np.arange(len(smoothdata2)), index=np.arange(len(smoothdata2)), columns=['target'])

        result = pd.concat([smoothdata1,smoothdata2], axis=0)
        rater = pd.concat([rater1, rater2], axis=0)
        target = pd.concat([target1, target2], axis=0)
        result = pd.concat([result, rater, target], axis = 1)
        icc = pg.intraclass_corr(data=result, ratings=col, raters='rater', targets= 'target')
        if(icc['ICC'][5]<0):
            icc['ICC'][5] = 0
        icc = pd.DataFrame([icc['ICC'][5]], columns = ["ICC"])
        icc_result = pd.concat([icc_result, icc], axis=0)
        r2 = pd.DataFrame([correlation(smoothdata1, smoothdata2)**2], columns = ["R2"])
        r2_result=pd.concat([r2_result, r2], axis=0)

    print("Finish!!!")

    if((r2_result<0.7).sum().sum()>1 or (icc_result<0.8).sum().sum()>1):
        print("This data is unacceptible.")
    else:
        print("This data is acceptible.")
    icc_resultv=icc_result.reset_index(drop=True)
    r2_result=r2_result.reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.margins(0)
    ax1.scatter(np.arange(1,len(r2_result)+1),r2_result)
    ax1.grid(linewidth=0.5, axis="y", c="black",linestyle="--")
    ax1.axhspan(0.7,1, facecolor='green', alpha=0.3, label="acceptible")
    ax1.axhspan(0,0.7, facecolor='red', alpha=0.3, label="unacceptible")
    ax1.set_xticks(np.arange(0, len(r2_result)+2))
    ax1.set_yticks(np.arange(0, 1, 0.1))
    ax1.set_title("R2")
    ax1.set_xlabel("Reference Data")
    ax1.legend()

    ax2.scatter(np.arange(1,len(icc_result)+1),icc_result)
    ax2.margins(0)
    ax2.grid(linewidth=0.5, axis="y", c="black",linestyle="--")
    ax2.axhspan(0.75,1, facecolor='green', alpha=0.3, label="acceptible")
    ax2.axhspan(0,0.75, facecolor='red', alpha=0.3, label="unacceptible")
    ax2.legend()
    ax2.set_xticks(np.arange(0, len(icc_result)+2))
    ax2.set_yticks(np.arange(0, 1, 0.1))
    ax2.set_xlabel("Reference Data")
    ax2.set_title("ICC")

    plt.show()