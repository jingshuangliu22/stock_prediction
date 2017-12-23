# -*- coding: utf-8 -*-

import numpy as np




def getFeatureData(dataSet):
    C = dataSet[:,3] # we use the colsing price to construct thecnical index based on price
    H = dataSet[:,1] # highest price
    L = dataSet[:,2] # lowest price
    O = dataSet[:,0] # open price
    Vol = dataSet[:,4] # volumn
    #Val = dataSet[:,5] # value
    
    return O,C,H,L,Vol
    
def SimNdaysMoav(C,n=10):
    m = len(C)
    SimNdaysMoav = np.zeros((m))
    for i in range((n-1),m):
        SimNdaysMoav[i]= sum(float(a) for a in C[(i-n+1):(i+1)])/n
    return SimNdaysMoav
    
def WeiNdaysMoav(C,n=10):
    m = len(C)
    WNMoav=0.0
    WeiNdaysMoav = np.zeros((m))
    for i in range((n-1),m):
        WNMoav = 0.0
        for j in range(1,(n+1)):
            WNMoav += j*C[i-n+j]
        WeiNdaysMoav[i] = WNMoav/sum(range(1,(n+1)))# 含前不含后
    return WeiNdaysMoav
    
def MomentumN(C,n=10):
    m = len(C)
    MomentumN = np.zeros((m))
    for i in range((n-1),m):
        MomentumN[i] = C[i]-C[i-n+1]
    return MomentumN
        
def StoKD(C,H,L,n=10):
    m = len(C)
    RSV = np.zeros(m,)
    StoK = np.ones((m))*50
    StoD = np.ones((m))*50
    for i in range((n-1),m):
        RSV[i] = 100.0*(C[i]-L[(i-n+1):(i+1)].min())/(H[(i-n+1):(i+1)].max()-L[(i-n+1):(i+1)].min())
        StoK[i] = (2.0/3)*StoK[i-1]+(1.0/3)*RSV[i]
        StoD[i] =  (2.0/3)*StoD[i-1]+(1.0/3)*StoK[i]
    return StoK,StoD
    
    
def LaWR(C,H,L,n=10):
    m = len(C)
    LaWR = np.zeros((m))
    for i in range((n-1),m):
        LaWR[i] = (H[(i-n+1):(i+1)].max()-C[i])*100.0/(H[(i-n+1):(i+1)].max()-L[(i-n+1):(i+1)].min())
    return LaWR
    
def OscillatorAD(H,L,C,n=10):
    m = len(C)
    OscAD = np.zeros((m))
    for i in range(m):
        OscAD[i] = (H[i]-C[i])/(H[i]-L[i])
        if(H[i]==L[i]):
            OscAD[i]=OscAD[i-1]
    return OscAD
    
def CCI(H,L,C,n=10):
    m = len(C)
    M =(H+C+L)/3
    SM = np.zeros((m))
    D = np.zeros((m))
    Abs = np.zeros((m))
    CCI = np.zeros((m))
    for i in range((n-1),m):
        SM[i] = sum(float(a) for a in M[(i-n+1):(i+1)])/n
        Abs[i] = abs(M[i-n+1]-SM[i])
    for j in range((n-1),m):
        D[j] = sum(float(a) for a in Abs[(j-n+1):(j+1)])/n # SM,D first 10 elements are not valide
        CCI[j] = (M[j]-SM[j])/(0.015*D[j]) # CCI first 10 elements are not valide
    return CCI
        
def RSI(O,C,H,L,n=10):
    m = len(C)
    RSI = np.zeros((m))
    RS = 0.0
    up = 0.0
    dw =0.0
    for i in range((n-1),m):
        up=0
        dw=0
        for j in range(n):
            if O[i-j]<C[i-j]:
                up = up + C[i-j] -O[i-j]
            else: dw = dw + O[i-j]-C[i-j]
        if (dw==0):
            RSI[i]=100
        else:
            RS = up/dw
            RSI[i] = 100.0*RS/(1+RS)
    return RSI
        
# # 起始始数据采用SH000300  2005年1月4日的数据,全为0
# SH0001 diff0 = 28.69 , diff1 = 28.38,dea0 = 35.86
def MACD(C,diff0 = 28.69 , diff1 = 28.38,dea0 = 35.86):# 此处计算出的MACD 实际为DEA，为DIF的9日移动均值，该处计算为参考论文02中的数据指标
    m = len(C)
    DIFF = np.zeros((m))
    DEA = np.zeros((m))
    EMA12 = np.zeros((m))
    EMA26 = np.zeros((m))
    MACDBar = np.zeros((m))
    DIFF[0]= diff0
    DIFF[1] = diff1
    DEA[0]=dea0
    EMA26[0] = (297.0/28.0)*(-(13.0/11.0)*(DIFF[1]-(28.0/351.0)*C[1]) + DIFF[0])
    EMA12[0] = DIFF[0] + EMA26[0]
    for i in range(1,m):
        EMA26[i] = (25.0/27.0)*EMA26[i-1] + (2.0/27.0)*C[i]
        EMA12[i] = (11.0/13.0)*EMA12[i-1] + (2.0/13.0)*C[i]
        DIFF[i] = EMA12[i]-EMA26[i]
        DEA[i] = 0.8*DEA[i-1]+0.2*DIFF[i]
    MACDBar = 2*(DIFF-DEA)
#    return EMA12,EMA26
    return DEA,DIFF,MACDBar

def rateVolBS(C,H,L,Vol):
    return (((C-L)-(H-C))/(H-L))*Vol    
    
def featureConstruction(dataSet):
    m = dataSet.shape[0]
    O,C,H,L,Vol = getFeatureData(dataSet)
    features = np.zeros((m,12))
    SNMoav = SimNdaysMoav(C,n=10)
    WNsMoav = WeiNdaysMoav(C,n=10)
    MN= MomentumN(C,n=10)
    StoK,StoD= StoKD(C,H,L,n=10)
    LWR= LaWR(C,H,L,n=10)
    OAD= OscillatorAD(H,L,C,n=10)
    CCI1= CCI(H,L,C,n=10)
    RSI1 = RSI(O,C,H,L,n=10)
    DEA,DIFF,MACDBar = MACD(C,diff0 = 28.69 , diff1 = 28.38,dea0 = 35.86)
    #rateVBS = rateVolBS(C,H,L,Vol)
    features[:,0] = SNMoav.transpose()
    features[:,1] = WNsMoav.transpose()
    features[:,2] = MN.transpose()
    features[:,3] = StoK.transpose()
    features[:,4] = StoD.transpose()
    features[:,5] = LWR.transpose()
    features[:,6] = OAD.transpose()
    features[:,7] = CCI1.transpose()
    features[:,8] = RSI1.transpose()
    features[:,9] = DEA.transpose()
    features[:,10] = DIFF.transpose()
    features[:,11] = MACDBar.transpose()
    #features[:,12] = rateVBS.transpose()
    return features


    
    
'''    SimNdaysMoav = SimNdaysMoav(C,n=10)
    WeiNdaysMoav = WeiNdaysMoav(C,n=10)
    MomentumN= MomentumN(C,n=10)
    StoK,StoD= StoKD(C,H,L,n=10)
    LaWR= LaWR(C,H,L,n=10)
    OscAD= OscillatorAD(H,L,C,n=10)
    CCI= CCI(H,L,C,n=10)
    RSI = RSI(O,C,H,L,n=10)
    DEA,DIFF,MACDBar = MACD(C,diff0 = 28.69 , diff1 = 28.38,dea0 = 33.16)
'''
