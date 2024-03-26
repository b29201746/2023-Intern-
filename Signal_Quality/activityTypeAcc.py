import numpy as np
from sgnUtility import resampleSGN
from time_cadence import cal_cadence_time
from scipy import signal

WINDOW = 8
FS = 25
BUFFER_SIZE = 5

CADENCE_LOWER_LIMIT = 60
CADENCE_UPPER_LIMIT = 240

ACT_TYPE_NONE = -2
ACT_TYPE_OTHER = -1
ACT_TYPE_STATIC = 0
ACT_TYPE_WALK = 1
ACT_TYPE_RUN = 2


def i_mode(array):
    array = sorted(array)
    tmp = array[0]
    mode_val = array[0]
    mode_count = 1
    count = 1
    for i in range(1, len(array)):
        if array[i] == tmp:
            count += 1
        if array[i] != tmp or i == len(array) - 1:
            if count >= mode_count:
                mode_count = count
                mode_val = tmp
            tmp = array[i]
            count = 1
    return mode_val, mode_count


class ActivityTypeAcc(object):
    def __init__(self):
        # activity type
        self.freqBuffer = [0 for _ in range(BUFFER_SIZE)]
        self.typeBuffer = [ACT_TYPE_NONE for _ in range(BUFFER_SIZE)]
        self.dominateActType = ACT_TYPE_NONE
        self.dominateFreq = 0
        # acc detection
        self.accXStd = [0 for _ in range(WINDOW)]
        self.accZMean = [0 for _ in range(WINDOW)]
        self.accNStd = [0 for _ in range(WINDOW)]
        self.accNMean = [0 for _ in range(WINDOW)]
        self.accNAmp = [0 for _ in range(WINDOW)]
        self.accN = [0 for _ in range(WINDOW * FS)]
        self.circularIdx = 0

    def runWalkDet(self, feat):
        feat1 = feat[0]
        feat2 = feat[1]
        feat3 = feat[2]
        feat4 = feat[3]
        runFreq = feat[4]
        walkFreq = feat[5]
        actType = ACT_TYPE_OTHER
        if feat2 > 113.37:
            if feat1 > 300 and runFreq > 120:
                actType = ACT_TYPE_RUN
            elif feat1 <= 300 and walkFreq < 140 and runFreq >= walkFreq:
                actType = ACT_TYPE_WALK
            else:
                actType = ACT_TYPE_OTHER
        if self.dominateActType == ACT_TYPE_WALK and self.dominateFreq > 0:
            if abs(self.dominateFreq - walkFreq) <= 9:
                actType = ACT_TYPE_WALK
        if self.dominateActType == ACT_TYPE_RUN and self.dominateFreq > 0:
            if abs(self.dominateFreq - runFreq) <= 9:
                actType = ACT_TYPE_RUN
        if abs(runFreq - walkFreq) < 9 and walkFreq < 140 and feat1 < 500:
            actType = ACT_TYPE_WALK
        if (feat3 < 150 and feat3 < feat4 * 0.7) or feat3 < 100:
            actType = ACT_TYPE_STATIC
        return actType

    def updateDom(self, actType, runFreq, walkFreq):
        freq = 0
        if actType == ACT_TYPE_RUN:
            freq = runFreq
        elif actType == ACT_TYPE_WALK:
            freq = walkFreq
        for i in range(BUFFER_SIZE - 1):
            self.freqBuffer[i] = self.freqBuffer[i + 1]
            self.typeBuffer[i] = self.typeBuffer[i + 1]
        self.freqBuffer[BUFFER_SIZE - 1] = freq
        self.typeBuffer[BUFFER_SIZE - 1] = actType

    def getDom(self):
        tmpArr = self.typeBuffer.copy()
        validCount = 0
        for actType in tmpArr:
            if actType != ACT_TYPE_NONE:
                validCount += 1
        if validCount >= 3:
            modeVal, modeCount = i_mode(tmpArr)
            if modeCount > (validCount / 2):
                self.dominateActType = modeVal
            else:
                self.dominateActType = ACT_TYPE_OTHER
        # find dominant (freq)
        tmpArr = self.freqBuffer.copy()
        validCount = 0
        for freq in tmpArr:
            if freq != 0:
                validCount += 1
        if validCount >= 3:
            modeVal, modeCount = i_mode(tmpArr)
            if modeCount >= (validCount / 2):
                validCount = 0
                sum = 0
                for i in range(BUFFER_SIZE):
                    if abs(self.freqBuffer[i] - modeVal) <= 15:
                        sum += self.freqBuffer[i]
                        validCount += 1
                if validCount == 0:
                    self.dominateFreq = 0
                else:
                    self.dominateFreq = sum / validCount
            else:
                self.dominateFreq = 0

    def update(self, accX, accY, accZ):  # 25 Hz
        accX = resampleSGN(accX, len(accX), FS)
        accY = resampleSGN(accY, len(accY), FS)
        accZ = resampleSGN(accZ, len(accZ), FS)
        # b, a = signal.butter(2, 12, btype="low", fs=FS)
        # accX = signal.lfilter(b, a, accX)
        # accY = signal.lfilter(b, a, accY)
        # accZ = signal.lfilter(b, a, accZ)
        accNorm = np.sqrt(np.square(accX) + np.square(accY) + np.square(accZ))
        # shift and update
        for i in range((WINDOW - 1) * FS):
            self.accN[i] = self.accN[i + FS]
        for i in range(FS):
            self.accN[(WINDOW - 1) * FS + i] = accNorm[i]
        self.accXStd[self.circularIdx] = np.std(accX)
        self.accZMean[self.circularIdx] = np.mean(accZ)
        self.accNStd[self.circularIdx] = np.std(accNorm)
        self.accNMean[self.circularIdx] = np.mean(accNorm)
        self.accNAmp[self.circularIdx] = max(accNorm) - min(accNorm)
        self.circularIdx += 1
        self.circularIdx %= WINDOW
        # get feature
        rCadence, _ = cal_cadence_time(self.accN, "run")
        wCadence, _ = cal_cadence_time(self.accN, "walk")
        if rCadence == None:
            rCadence = 0
        if wCadence == None:
            wCadence = 0
        if rCadence >= CADENCE_LOWER_LIMIT and rCadence <= CADENCE_UPPER_LIMIT:
            runFreq = rCadence
        else:
            runFreq = 0
        if wCadence >= CADENCE_LOWER_LIMIT and wCadence <= CADENCE_UPPER_LIMIT:
            walkFreq = wCadence
        else:
            walkFreq = 0
        featNormMean = np.mean(self.accNMean)
        featZMean = np.mean(self.accZMean)
        featXStd = np.mean(self.accXStd)
        featNormStd = np.mean(self.accNStd)
        featAmpMin = min(self.accNAmp)
        featAmpStd = np.std(self.accNAmp)
        if walkFreq == 0 and self.dominateActType == ACT_TYPE_WALK:
            if abs(self.dominateFreq - runFreq) < 9:
                walkFreq = runFreq
            elif abs(self.dominateFreq - runFreq / 2) < 9:
                walkFreq = runFreq / 2
            else:
                walkFreq = self.dominateFreq
        if featNormStd < 150 and walkFreq > 130:
            walkFreq /= 2
        # run walk detect
        feat = [featNormStd, featXStd, featAmpMin, featAmpStd, runFreq, walkFreq]
        actType = self.runWalkDet(feat)

        # update dominant
        self.updateDom(actType, runFreq, walkFreq)
        # find dominant
        self.getDom()

        return self.dominateActType, self.dominateFreq

