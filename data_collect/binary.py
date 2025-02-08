'''
This file is writen by Ming Yang
  for reading binary files of vecs format
'''

import numpy as np
import struct

def fbin_read(fname):
  data = np.fromfile(fname, dtype='int32')
  n, d = data[0], data[1]
  n, d = max(n, 1), max(d, 1)
  return data[2:].reshape(n, d).view('float32')

def ibin_read(fname):
  data = np.fromfile(fname, dtype='int32')
  n, d = data[0], data[1]
  return data[2:].reshape(n, d)

def ibin_save(filename, data):
  with open(filename, 'wb') as fp:
    fp.write(struct.pack('II', data.shape[0], data.shape[1]))
    for y in data:
      for x in y:
        a = struct.pack('I', x)
        fp.write(a)

def fbin_save(filename, data):
  with open(filename, 'wb') as fp:
    fp.write(struct.pack('II', data.shape[0], data.shape[1]))
    for y in data:
      for x in y:
        a = struct.pack('f', x)
        fp.write(a)


def ivecs_read(fname):
  a = np.fromfile(fname, dtype='int32')
  d = a[0]
  return a.reshape(-1, d + 1)[:, 1:].copy()
 
def fvecs_read(fname):
  return ivecs_read(fname).view('float32')


def ivecs_save(filename, data):
  with open(filename, 'wb') as fp:
    for y in data:
      d = struct.pack('I', y.size)
      fp.write(d)
      for x in y:
        a = struct.pack('I', x)
        fp.write(a)

def fvecs_save(filename, data):
  with open(filename, 'wb') as fp:
    for y in data:
      d = struct.pack('I', y.size)
      fp.write(d)
      for x in y:
        a = struct.pack('f', x)
        fp.write(a)
  