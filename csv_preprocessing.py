import os
import csv
import shutil
import cPickle as pkl
import numpy as np

if __name__ == '__main__':
    csvfile = file("stock_trend.csv","rb")
    reader = csv.reader(csvfile)
    dicts = {}
    for line in reader:
        dicts[line[0]] = [line[1],line[2]]
    csvfile.close()
    files1 = os.listdir("train/pos/")
    files2 = os.listdir("train/neg/")
    files3 = os.listdir("Development/pos/")
    files4 = os.listdir("Development/neg/")
    files5 = os.listdir("test/pos/")
    files6 = os.listdir("test/neg/")
    csvfile = file("stock_trend_new.csv","wb")
    writer = csv.writer(csvfile)
    for f in files1:
        writer.writerow([f,dicts[f][0],dicts[f][1]])
    for f in files2:
        writer.writerow([f,dicts[f][0],dicts[f][1]])
    for f in files3:
        writer.writerow([f,dicts[f][0],dicts[f][1]])
    for f in files4:
        writer.writerow([f,dicts[f][0],dicts[f][1]])
    for f in files5:
        writer.writerow([f,dicts[f][0],dicts[f][1]])
    for f in files6:
        writer.writerow([f,dicts[f][0],dicts[f][1]])
    csvfile.close()

