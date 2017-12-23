'''
construct the 10 features for the s&p index
'''
import numpy as np
import cPickle as pkl
import numpy as np
import csv
from featureConstruction import featureConstruction

def dumpindex():
    csvfile = file("stock_trend_new.csv","rb")
    reader = csv.reader(csvfile)
    stock = []
    for line in reader:
        tmp=[]
        print line
        tmp.append(float(line[2].replace(",","")))
        tmp.append(float(line[3].replace(",","")))
        tmp.append(float(line[4].replace(",","")))
        tmp.append(float(line[5].replace(",","")))
        tmp.append(float(line[6].replace(",","")))
        stock.append(tmp)
    csvfile.close()

    close_st = []
    high_st = []
    low_st = []
    y = [] #trend

    for i in range(len(stock)):
        close_st.append(stock[i][3])
        high_st.append(stock[i][1])
        low_st.append(stock[i][2])
    #construct the features
    st_fea = np.array(stock)
    st_fea = featureConstruction(st_fea)
    max=[]
    min=[]
    vol=[]
    for i in range(len(st_fea[0])):
        max.append(st_fea[9:,i].max())
        min.append(st_fea[9:,i].min())
        vol.append(max[i]-min[i])

    for i in range(len(st_fea)):
        for j in range(len(st_fea[0])):
            st_fea[i][j] = (st_fea[i][j]-min[j])*10000/vol[j]

    st_fea = st_fea[9:-1]
    print len(st_fea)

    #predict the close trend

    for i in range(10,len(close_st)):
        if close_st[i-1] > close_st[i]:
            y.append(0)
        else:
            y.append(1)
    print len(y)
    '''
    #predict the high price trend
    for i in range(20,len(high_st)):
        if high_st[i-1] > high_st[i]:
            y.append(0)
        else:
            y.append(1)
    '''

    len_train = 1425-9
    len_val = 169
    print len(st_fea[len_train+len_val:])
    #dump 10feature-close
    f = open("stock_feature_close.pkl","wb")
    pkl.dump((st_fea[:len_train],y[:len_train]),f,-1)
    pkl.dump((st_fea[len_train:len_train+len_val],y[len_train:len_train+len_val]),f,-1)
    pkl.dump((st_fea[len_train+len_val:],y[len_train+len_val:]),f,-1)
    f.close()
    sum=0

if __name__ == '__main__':
    dumpindex()