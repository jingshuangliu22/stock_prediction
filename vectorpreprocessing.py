import os
import csv
import shutil
import cPickle as pkl
import numpy as np

if __name__ == '__main__':
    #extract the amount of information for every day
    f = open('stock_ins_num.pkl', 'rb')
    train_pos,train_neg = pkl.load(f)
    dev_pos,dev_neg = pkl.load(f)
    test_pos,test_neg = pkl.load(f)
    f.close()
    len_train_pos = 0
    len_train_neg = 0
    for i in train_pos:
        len_train_pos += i
    for i in train_neg:
        len_train_neg += i
    print len_train_pos+len_train_neg
    #extract the news vectors for every day
    f2 = open('stock_vecs_dm.pkl','rb')
    train_dev_vec_pos,train_dev_vec_neg = pkl.load(f2)
    test_vec_pos,test_vec_neg = pkl.load(f2)
    f2.close()
    train_vec_pos = train_dev_vec_pos[:len_train_pos]
    dev_vec_pos = train_dev_vec_pos[len_train_pos:]
    train_vec_neg = train_dev_vec_neg[:len_train_neg]
    dev_vec_neg = train_dev_vec_neg[len_train_neg:]


    #compute the average vector for every day
    print "computing average vector..."
    #train_pos
    idx = 0
    train_avg_pos = []
    find = 0
    for i in train_pos:
        tmp = np.zeros(train_vec_pos[0].shape)
        for j in range(0,i):
            tmp += train_vec_pos[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        train_avg_pos.append(tmp)

    #train_neg
    idx = 0
    train_avg_neg = []
    find = 0
    for i in train_neg:
        tmp = np.zeros(train_vec_neg[0].shape)
        for j in range(0,i):
            tmp += train_vec_neg[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        train_avg_neg.append(tmp)

    #dev_pos
    idx = 0
    dev_avg_pos = []
    find = 0
    for i in dev_pos:
        tmp = np.zeros(dev_vec_pos[0].shape)
        for j in range(0,i):
            tmp += dev_vec_pos[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        dev_avg_pos.append(tmp)

    #dev_neg
    idx = 0
    dev_avg_neg = []
    find = 0
    for i in dev_neg:
        tmp = np.zeros(dev_vec_neg[0].shape)
        for j in range(0,i):
            tmp += dev_vec_neg[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        dev_avg_neg.append(tmp)

    #test_pos
    idx = 0
    test_avg_pos = []
    find = 0
    for i in test_pos:
        tmp = np.zeros(test_vec_pos[0].shape)
        for j in range(0,i):
            tmp += test_vec_pos[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        test_avg_pos.append(tmp)

    #test_neg
    idx = 0
    test_avg_neg = []
    find = 0
    for i in test_neg:
        tmp = np.zeros(test_vec_neg[0].shape)
        for j in range(0,i):
            tmp += test_vec_neg[idx]
            idx += 1
        if i != 0:
            tmp = tmp/i
        test_avg_neg.append(tmp)

    print "finished"
    #dump average vector
    f = open('stock_avg_vecs.pkl', 'wb')


    pkl.dump((train_avg_pos,train_avg_neg), f, -1)
    pkl.dump((dev_avg_pos,dev_avg_neg), f, -1)
    pkl.dump((test_avg_pos,test_avg_neg), f, -1)
    f.close()


