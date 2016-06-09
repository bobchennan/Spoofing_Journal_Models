from __future__ import print_function
import numpy as np
import lasagne
import cPickle
import sys
import os
from htk import HTKFeat_read
from sklearn.preprocessing import normalize
import theano
import theano.tensor as T
from keras.models import *# model_from_json
import math
from keras import backend as K
def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

last_file = None
data = None
FEATURE_EX  = 15
def parse(fea_str):
    s = fea_str.split('=')
    t = s[1].split('[')
    if len(t)>1:
        u = t[1].split(',')
        v = u[1].split(']')
    else:
        u = 0
        v = sys.maxint
    return s[0], t[0], int(u[0]), int(v[0])

def load_data(scp):
    global last_file
    for i in open(scp, "r"):
        name, file, start, end = parse(i)
        if file!=last_file:
            last_file = file
            data = HTKFeat_read(file).getall()
        value=[]
        for j in xrange(start-FEATURE_EX,end-FEATURE_EX+1):
            value.append(data[j:j+2*FEATURE_EX+1].ravel())
        yield name, np.array(value)

def get_output(fn, inputs, norm):
    #return model.predict({'input':np.array([norm.transform(inputs)])})['output'][0]
    #return fn([np.array(norm.transform(inputs))])[0]
    return fn([norm.transform(inputs)])[0]

if len(sys.argv)!=3:
    print("Usage: ../run.sh write.py model scp")
    exit(0)
layer_idx=-9
with open(sys.argv[1], "rb") as f:
    norm = cPickle.load(f)
    model = cPickle.load(f)#.replace('"activation": "sigmoid"','"activation": "linear"')
  #  model = model_from_json(model)
  #  model.load_weights('sigmoid.h5') 
if type(model)==Sequential:
    warning("totally "+str(len(model.layers))+" layers"+" used "+str(layer_idx)+"th")
output=K.function([model.layers[0].input],
                      [model.layers[layer_idx].get_output(train=False)])
#output=K.function([model.layers[0].input],[model.layers[0].input])
#output=K.function([model.inputs[i].input for i in model.input_order],
#                      [model.nodes['l4'].get_output(train=False)])
for i in load_data(sys.argv[2]):
    name, value = i
    value = get_output(output, inputs=value, norm=norm)
    #print(value)
    #break
    #print(normalize(value,'l2'))
    #print(normalize(value,'l2')*math.sqrt(value.shape[1]))
    #value = get_output(model, inputs=value, norm=norm)
    #value=normalize(value,'l2')*math.sqrt(value.shape[1])
    value=np.concatenate((value.mean(axis=0),value.std(axis=0)))
    #print(value.shape)
    print(name+" "+(" ".join(map(str, value))))
    #break
