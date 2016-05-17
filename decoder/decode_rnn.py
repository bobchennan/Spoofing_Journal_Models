from __future__ import print_function
import numpy as np
import cPickle
import sys
import os
from htk import HTKFeat_read
from sklearn.preprocessing import normalize
import theano
import theano.tensor as T
from keras.models import model_from_json
#from keras import backend as K
def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

last_file = None
data = None
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
        yield name, np.array(data[start:end])

def apply_to_zeros(lst):
    inner_max_len = max(map(lambda x:x.shape[0], lst))
    result = np.zeros([len(lst), inner_max_len, lst[0].shape[1]], lst[0].dtype)
    for i, frame in enumerate(lst):
        for j, row in enumerate(frame):
            for k, val in enumerate(row): 
                result[i][j][k] = val
    return result

def get_output(fn, inputs):
    #return fn.predict(np.array([norm.transform(inputs)]))
    #return fn.predict({'input':np.array([norm.transform(inputs)])})['output'][0]
    #return fn([np.array([norm.transform(inputs)])])[0][0]
    return fn(apply_to_zeros(inputs))[0]

if len(sys.argv)!=3:
    print("Usage: ../run.sh write.py model scp")
    exit(0)
layer_idx=-14
with open(sys.argv[1], "rb") as f:
    norm = cPickle.load(f)
    model = cPickle.load(f)#.replace('"activation": "relu"','"activation": "linear"')
    #model = model_from_json(model)
    #model.load_weights('lstm.h5') 
output=theano.function([model.inputs[i].input for i in model.input_order],
                      [model.nodes['dropout'].get_output(train=False)])
names, values= [], []
for i in load_data(sys.argv[2]):
    name, value = i
    names.append(name)
    values.append(norm.transform(value))
    #value = get_output(model, inputs=value, norm=norm)
    if len(names)==128:
        value = get_output(output, inputs=values)
        for j in xrange(len(names)):
            print(names[j]+" "+(" ".join(map(str, value[j]))))
        names, values=[], []
if len(names)>0:
    value = get_output(output, inputs=values)
    for j in xrange(len(names)):
        print(names[j]+" "+(" ".join(map(str,value[j]))))
