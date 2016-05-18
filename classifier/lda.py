#!/bin/bash
from __future__ import print_function
import sys
import numpy as np
from sklearn.lda import LDA
from sklearn import svm
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import math
def warning(*objs):
    print(*objs, file=sys.stderr)
norm = None
def read_fea(filename, test=False):
    out=open(filename,"r")
    X=[]
    Y=[]
    for i in out:
        s=i.rstrip().split(" ")
        d=int(s[0])
        tmp=np.array([float(j) for j in s[1:len(s)]])
        X.append(tmp)
        if test:
            Y.append(d)
        else:
            Y.append(d)#1 if d==0 else 0)
    return normalize(np.array(X),norm='l2'),Y

def sample_data(X, Y, value=0):
    XX=[]
    for i in xrange(len(Y)):
        if Y[i]==value:
            XX.append(X[i])
    return XX

out=open(sys.argv[1],"r")
model=LDA()
X, Y = read_fea(sys.argv[1])
sel = VarianceThreshold(threshold=0)
model.fit(sel.fit_transform(X), Y)
warning("useful features dim: "+str(len(sel.get_support(True))))
if hasattr(model,'score'):
    warning("accuracy on training set: "+str(model.score(sel.transform(X), Y)))
    if len(sys.argv)>2:
        X, Y = read_fea(sys.argv[2])
        warning("accuracy on cv set: "+str(model.score(sel.transform(X), Y)))

    if len(sys.argv)>3:
        X, Y = read_fea(sys.argv[3])
        warning("accuracy on dev set: "+str(model.score(sel.transform(X), Y)))

if len(sys.argv)>4:
    ref = model.decision_function(sel.transform(X))
    X, Y = read_fea(sys.argv[4], True)
    Z = model.decision_function(sel.transform(X))
    Z = (Z-ref.mean(axis=0)[np.newaxis,:])/ref.std(axis=0)[np.newaxis,:]
    for i in xrange(len(Y)):
        ZZ=np.array(Z[i][1:])
        print('S'+str(Y[i])+' '+str(Z[i][0]))
