#!/usr/bin/env python

"""
Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)

With n=5, i.e. 32-layer network from the paper, this achieves a validation error of 6.88% (vs 7.51% in the paper).
The accuracy has not yet been tested for the other values of n.
"""

from __future__ import print_function

import sys
import os
sys.setrecursionlimit(10000)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
import time
import string
import random
import cPickle

import numpy as np
np.random.seed(721)
import theano
import theano.tensor as T
from htk import HTKFeat_read
from sklearn.preprocessing import StandardScaler
### config ###
from keras.models import Graph
from keras.layers.core import Masking, Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM 
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

FEATURE_DIM = 48
HIDDEN_SIZE = 512
OUTPUT_SIZE = 6
#FILES_SHOW = 5000
data = None
LSTM_STEP = 70

def parse(fea_str, prefix):
    s = fea_str.split('=')
    t = s[1].split('[')
    u = t[1].split(',')
    v = u[1].split(']')
    return prefix+"_"+s[0], t[0], int(u[0]), int(v[0])

def load_data(scp, mlf, prefix="train"):
    global data
    tr = {}
    for i in open(mlf,"r"):
        j = i.rstrip().split('\t')
        tr[prefix+"_"+j[0]] = int(j[1])
    file_data = {}
    X = []
    Y = []
    for i in open(scp,"r"):
        name, file, start, end = parse(i, prefix)
        end = max(end, start+LSTM_STEP)
        #if not name in X.keys():
        #    X[name]=[]
        #    Y[name]=[]
        if not file in file_data.keys():
            file_data[file]=HTKFeat_read(file).getall()
        if data is None:
            curr = 0
            data = file_data[file][start:end+1]
        else:
            curr = data.shape[0]
            data = np.vstack((data, file_data[file][start:end+1]))
        #print(str(curr)+":"+str(start)+","+str(end))
        #X.extend(range(curr, curr+end-start-LSTM_STEP+1))
        if prefix=="train":
            X.extend(range(curr, curr+end-start-LSTM_STEP+1))
            Y.extend([tr[name] for i in range(end-start-LSTM_STEP+1)])
            #print(str(X))
        else:
            X.append((curr, curr+end-start+1))
            Y.extend([tr[name]])
    #del file_data
    return X, Y

# ##################### Build the neural network model #######################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    if shuffle:
        inds=range(len(inputs))
        np.random.shuffle(inds)
        inputs=[inputs[j] for j in inds]
        targets=[targets[j] for j in inds]
    #nt=FILES_NUM
    #files=[HTKFeat_read(filename=file_pos[names[i]]) for i in xrange(FILES_NUM)]
    #file_names=[names[i] for i in xrange(FILES_NUM)]
    #inds=[0 for i in xrange(FILES_NUM)]
    #inds=[0 for i in xrange(len(names))]
    #avail=Set([i for i in xrange(len(inputs))])
    curr = 0
    while True:
        X=[]
        Y=[]
        finished=False
        for r in xrange(batchsize):
            if len(inputs)==curr:
                finished=True
                break
            idx=curr
            curr+=1
            #files[idx].seek(inputs[file_names[idx]][inds[idx]])
            #XX=[]
            #for _ in xrange(FEATURE_EX):
            #    XX.extend(files[idx].next().tolist())
            if type(inputs[idx])==int:
                XX=data[inputs[idx]:inputs[idx]+LSTM_STEP]
            else:
                XX=data[inputs[idx][0]:inputs[idx][1]]
            #XX=data[inputs[names[idx]][inds[idx]]:inputs[names[idx]][inds[idx]]+2*FEATURE_EX+1].ravel()
            #if np.isnan(np.sum(XX)):
            #    print(names[idx]+":"+str(inputs[names[idx]][inds[idx]])+","+str(inputs[names[idx]][inds[idx]]+FEATURE_EX))
            X.append(XX)
            Y.append(targets[idx])
            #Y.append(targets[names[idx]][inds[idx]])
            #inds[idx]+=1
            #if inds[idx]>=len(inputs[names[idx]]):
            #   avail.remove(idx)
               #files[idx]=None
               #file_names[idx]=None
               #inds[idx]=0
        #print(str(X.mean(axis=1)))
        if len(X)>0:
            yield np.array(X), np_utils.to_categorical(Y, nb_classes=6)
        if finished:
            break

# ############################## Main program ################################

def main(n=5, num_epochs=9):
    model = Graph()
    model.add_input(name='input', input_shape=(None,FEATURE_DIM))
    model.add_node(Masking(0.0),name='mask',input='input')
    model.add_node(TimeDistributedDense(HIDDEN_SIZE),name='transform',input='mask')
    model.add_node(LSTM(256+64,return_sequences=True),name='forward',input='transform')
    model.add_node(LSTM(256+64),name='backward',input='forward')
    model.add_node(Dense(1024, activation='sigmoid'),name='representation',input='backward')
    model.add_node(Dropout(0.5),name='dropout',input='representation')
    model.add_node(Dense(OUTPUT_SIZE, activation='softmax'),name='softmax',input='dropout')
    model.add_output(name='output',input='softmax')
    model.compile('adam', {'output': 'categorical_crossentropy'})
    model.summary()
    global data
    # Load the dataset
    print("Loading data...")
    try:
        with open("spoof3.pickle", "rb") as f:
            tmp = cPickle.load(f)
            if tmp!=LSTM_STEP:
                print("Context window don't match.")
                raise ValueError,'invalid window'
            data, X_train, Y_train, X_test, Y_test = cPickle.load(f)
    except:
        print("Regenerate data!")
        X_train, Y_train = load_data("spoof_train.scp", "mlf")
        X_test, Y_test = load_data("spoof_dev.scp", "mlf","dev")
        with open("spoof3.pickle", "wb") as f:
            cPickle.dump(LSTM_STEP, f)
            cPickle.dump([data, X_train, Y_train, X_test, Y_test], f)
    # Normalize data
    #data = StandardScaler().fit_transform(data)
    norm = StandardScaler().fit(data[:np.array(X_train).max()])
    data = norm.transform(data)
    print("Starting training...")
    # We iterate over epochs:
    with open("lstm.pickle", "wb") as f:
        cPickle.dump(norm, f)
        cPickle.dump(model.to_json(), f)
    best_record=0
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, 1024, shuffle=True, augment=True):
            inputs, targets = batch
            tmp = model.train_on_batch({'input':inputs, 'output':targets})
            train_err += tmp[0]
            if len(tmp)>1:
                print(str(tmp))
            train_batches+=1
        #print("Epoch "+str(epoch)+" Accuracy:"+str(train_err/train_batches))

        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 1, shuffle=False):
            inputs, targets = batch
            value = model.predict({'input':inputs})
            if len(value.keys())==0:
                print(inputs)
                print(inputs.shape)
            value = np.array(value['output'])
            #err, acc = model.test_on_batch(inputs, targets, accuracy=True)
            test_acc += np.sum(value.argmax(axis=1)==targets.argmax(axis=1))
            test_batches += value.shape[0]
        if 1.0*test_acc/test_batches>best_record:
            best_record=1.0*test_acc/test_batches
            model.save_weights('lstm.h5',overwrite=True)
            ch = '*'
        else:
            ch = '-'
        print("Epoch "+str(epoch)+" results:")
        print("  test accuracy:\t\t{:.2f} % {:s}".format(
            1.0 * test_acc / test_batches * 100,ch))
    # dump the network weights to a file :
    #
    # And load them again later on like this:
    # with np.load('cifar10_deep_residual_model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
        print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N [EPOCHS]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("EPOCHS: number of training epochs to perform (default: 82)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[3])
        main(**kwargs)
