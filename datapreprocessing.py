import os
import csv
import shutil
import cPickle as pkl
allFileNum = 0
def printPath(path):
    tmp = []
    files = os.listdir(path)
    #fw = open("no_class.txt","w")
    for f in files:
        ff = open(path+"/"+f,"r")
        i=0
        for each_line in ff:
            i+=1
        tmp.append(i)
        ff.close
    #fw.close
    #print tmp
    #print len(tmp)
    return tmp




if __name__ == '__main__':

    f = open('doc_vector.pkl', 'wb')

    train_pos = printPath('D:/BUAA/2017/ReutersNews/Training/pos')
    train_neg = printPath('D:/BUAA/2017/ReutersNews/Training/neg')
    dev_pos = printPath('D:/BUAA/2017/ReutersNews/Development/pos')
    dev_neg = printPath('D:/BUAA/2017/ReutersNews/Development/neg')
    test_pos = printPath('D:/BUAA/2017/ReutersNews/Test/pos')
    test_neg = printPath('D:/BUAA/2017/ReutersNews/Test/neg')
    pkl.dump(train_pos+train_neg, f, -1)
'''
    pkl.dump((train_pos,train_neg), f, -1)
    pkl.dump((dev_pos,dev_neg), f, -1)
    pkl.dump((test_pos,test_neg), f, -1)
    f.close()

    
    f = open('stock_ins_num.pkl', 'rb')
    train_pos,train_neg = pkl.load(f)
    dev_pos,dev_neg = pkl.load(f)
    test_pos,test_neg = pkl.load(f)
    f.close()

    train_new_pos =train_pos + dev_pos
    print len(train_new_pos)
    print train_new_pos'''