'''
1.make the whole matrix
2.make the matrix by week
3.make the idx by week
'''
import os
import csv
import shutil
import cPickle as pkl
import numpy as np
import cPickle as pkl

def dumpmatrix():
    csvfile = file("stock_trend_new.csv","rb")
    reader = csv.reader(csvfile)
    stock = []
    for line in reader:
        stock.append(line[1])
    csvfile.close()
    new_matrix = []
    new_matrix_label = []
    f = open("stock_avg_vecs.pkl", 'rb')
    train_pos,train_neg = pkl.load(f)
    dev_pos,dev_neg = pkl.load(f)
    test_pos,test_neg = pkl.load(f)
    f.close()

    p_pos = 0
    p_neg = 0

    for i in range(1425):
        if stock[i] == '1':
            new_matrix.append(train_pos[p_pos])
            new_matrix_label.append(1)
            p_pos += 1
        else:
            new_matrix.append(train_neg[p_neg])
            new_matrix_label.append(0)
            p_neg += 1
    p_pos = 0
    p_neg = 0
    for i in range(1425,1425+169):
        if stock[i] == '1':
            new_matrix.append(dev_pos[p_pos])
            new_matrix_label.append(1)
            p_pos += 1
        else:
            new_matrix.append(dev_neg[p_neg])
            new_matrix_label.append(0)
            p_neg += 1
    p_pos = 0
    p_neg = 0
    for i in range(1425+169,1425+169+190):
        if stock[i] == '1':
            new_matrix.append(test_pos[p_pos])
            new_matrix_label.append(1)
            p_pos += 1
        else:
            new_matrix.append(test_neg[p_neg])
            new_matrix_label.append(0)
            p_neg += 1

    f = open("stock_whole_vecs.pkl", 'wb')

    pkl.dump(new_matrix,f,-1)
    pkl.dump(new_matrix_label,f,-1)
    f.close()

def matrixbyweek():

    f = open("stock_whole_vecs.pkl", 'rb')
    matrix = pkl.load(f)
    label = pkl.load(f)
    f.close()
    x_train = []
    y_train = label[6:1425]
    x_dev = []
    y_dev = label[1425+6:1425+169]
    x_test = []
    y_test = label[1425+169+6:]

    for i in range(1425-6):
        tmp = []
        for j in range(7):
            tmp.append(matrix[i+j])
        x_train.append(tmp)

    for i in range(1425,1425+169-6):
        tmp = []
        for j in range(7):
            tmp.append(matrix[i+j])
        x_dev.append(tmp)

    for i in range(1425+169,1425+169+190-6):
        tmp = []
        for j in range(7):
            tmp.append(matrix[i+j])
        x_test.append(tmp)

    f = open("stock_week_vecs.pkl", 'wb')
    pkl.dump((x_train,y_train),f,-1)
    pkl.dump((x_dev,y_dev),f,-1)
    pkl.dump((x_test,y_test),f,-1)
    f.close()

def indexbyweek():
    csvfile = file("stock_trend_new.csv","rb")
    reader = csv.reader(csvfile)
    label = []
    index = []
    for line in reader:
        label.append(line[1])
        index.append(float(line[2].replace(",",""))*10)
    csvfile.close()
    x_train = []
    y_train = label[6:1425]
    x_dev = []
    y_dev = label[1425+6:1425+169]
    x_test = []
    y_test = label[1425+169+6:]

    for i in range(1425-6):
        tmp = []
        for j in range(7):
            tmp.append(int(index[i+j]))
        x_train.append(tmp)

    for i in range(1425,1425+169-6):
        tmp = []
        for j in range(7):
            tmp.append(int(index[i+j]))
        x_dev.append(tmp)

    for i in range(1425+169,1425+169+190-6):
        tmp = []
        for j in range(7):
            tmp.append(int(index[i+j]))
        x_test.append(tmp)


    f = open("stock_week_idxs.pkl", 'wb')
    pkl.dump((x_train,y_train),f,-1)
    pkl.dump((x_dev,y_dev),f,-1)
    pkl.dump((x_test,y_test),f,-1)
    f.close()



if __name__ == '__main__':
    #dumpmatrix()
    #matrixbyweek()
    indexbyweek()
