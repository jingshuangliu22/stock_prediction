#!/usr/bin/python
import sys
import numpy as np
import gensim
from random import shuffle
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.model_selection import train_test_split

LabeledSentence = gensim.models.doc2vec.LabeledSentence
pos_file, neg_file, unsup_file = "data/pos.txt","data/neg.txt","data/pos.txt"
def get_dataset():
    print "loading data..."
    with open(pos_file,'r') as infile:
        pos_reviews = infile.readlines()
    with open(neg_file,'r') as infile:
        neg_reviews = infile.readlines()
    with open(unsup_file,'r') as infile:
        unsup_reviews = infile.readlines()
    #print np.array(pos_reviews)
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    print "spliting data..."
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
    
    
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z.lower().replace('\n','') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        #treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s '%c) for z in corpus]
        corpus = [z.split() for z in corpus]
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    unsup_reviews = cleanText(unsup_reviews)

    
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    return x_train,x_test,unsup_reviews,y_train, y_test


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def train(x_train,x_test,unsup_reviews,size = 400,epoch_num=10):
    
    #model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=1)

    print "training..."
    #model_dm.build_vocab(x_train+x_test+unsup_reviews)
    model_dbow.build_vocab(x_train+x_test+unsup_reviews)
    all_train_reviews = x_train
    for epoch in range(epoch_num):
        #shuffle(all_train_reviews)
		#model_dm.train(all_train_reviews)
        model_dbow.train(all_train_reviews)
    for epoch in range(epoch_num):
        #shuffle(x_test)
        #model_dm.train(x_test)
        model_dbow.train(x_test)

    return model_dbow
    #return model_dm	

def get_vectors(model_dbow):
    print "geting vectors..."
    size,epoch_num = 300,100
    #train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    #train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    
    #test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    #test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs_dbow,test_vecs_dbow

def Classifier(train_vecs,y_train,test_vecs, y_test):
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)

    return lr
if __name__ == "__main__":
    size,epoch_num = 300,10
    x_train,x_test,unsup_reviews,y_train, y_test = get_dataset()
    
    model_dbow = train(x_train,x_test,unsup_reviews,size,epoch_num)
    train_vecs,test_vecs = get_vectors(model_dbow)
    lr=Classifier(train_vecs,y_train,test_vecs, y_test)  