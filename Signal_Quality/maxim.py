
import numpy as np
import pandas as pd
import dataDecode
import math
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt

def get_data_complement(signal):
    np_signal=np.array(signal)
    for i in range(len(np_signal)):
        if np_signal[i]<32768:
            np_signal[i]+=65535
    np_signal-=65535      
    return np_signal

time = "230728a"

fname= time + ".6C8"
Raw_Data=open(fname, "rb").read()
Data,sampling_rate,timestr=dataDecode.dataDecode.rawdataDecode(Raw_Data)


ACC_X = Data[0]
ACC_Y = Data[1] 
ACC_Z = Data[2]
PPG = Data[3]

ACC_X = pd.Series(ACC_X)
ACC_Y = pd.Series(ACC_Y)
ACC_Z = pd.Series(ACC_Z)
PPG = pd.Series(PPG)

Datalist = pd.concat([ACC_X, ACC_Y, ACC_Z, PPG], axis=1)
column_name =['ACC_X', 'ACC_Y', 'ACC_Z', 'PPG']
Datalist.columns = column_name

csv_name = time +'.csv'
Datalist.to_csv(csv_name, index = None)

data_duration=len(Data[0])/sampling_rate[0]

end_record_time=datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S') + timedelta(seconds = int(data_duration)) 
print('start record time: ',timestr)
print('end record time: ',datetime.strftime(end_record_time, '%Y-%m-%d %H:%M:%S'))
print('Record time: ', str(timedelta(seconds = int(data_duration)) ))

'''
# if need transfer x-axis to datetime, using code below
start_time = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
time_array = [start_time + timedelta(seconds=(1 / sampling_rate[0] * dot)) for dot in range(len(Data[0]))]
'''

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax2.plot(Data[0], label='ACC_X')
ax2.plot(Data[1], label='ACC_Y')
ax2.plot(Data[2], label='ACC_Z')
ax1.plot(Data[3], label='PPG')

ax1.grid(), ax1.legend()
ax2.grid(), ax2.legend()

ax1.set_xlim(0, len(Data[0]))
ax2.set_xlim(0, len(Data[1]))
plt.title(csv_name)
plt.show()
# np_ECG=get_data_complement(Data[0])

#%%

# import csv

# with open('Maxim.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for value in Data[0]:
#            writer.writerow([value])
#     for value in Data[1]:
#            writer.writerow([value])
#     for value in Data[2]:
#            writer.writerow([value])
#     for value in Data[0]:
#            writer.writerow([value])


