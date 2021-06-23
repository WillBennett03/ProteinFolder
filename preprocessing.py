"""
########################
### preprocessing.py ###
########################

~ Will Bennett 17/06/2021

Turns the proteinnet dataset into usable data for the model to make predictions from and train form.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python import tf2
from tensorflow.tools.docs.doc_controls import _FOR_SUBCLASS_IMPLEMENTERS
import tokeniser
import pickle
import os

MAX_SEQ_LEN = 20000

def getPolypeptideTokens(PolypeptidesIter): #takes a list of polypeptides and returns a list of polypeptides with amino acids represented by tokens e.g [0,0,0,1...] with a len of 20
    df_, dict_ = tokeniser.get_aminoacids()
    x = [tokeniser.GetVectoredPolypeptide(x, dict_) for x in PolypeptidesIter]
    return x

def getRecord(filename): #returns the data from proteinnet file
    df_, dict_ = tokeniser.get_aminoacids()
    file = open(filename, 'r')#gets txt file
    stage = ""
    primarys = np.array([])
    tertiarys = np.array([])
    masks = np.array([])
    current = [] #for stage where multiple lines are used
    skip = False

    #Loops over the line sin proteinnet file
    for line in file:
        primary_padded = np.zeros((MAX_SEQ_LEN, 20))
        tertiary_padded = np.zeros((MAX_SEQ_LEN,3,3))
        mask_padded = np.zeros(MAX_SEQ_LEN)

        if line[0] == '[':
            stage = line

        elif stage == "[PRIMARY]\n" and skip != True:
            AA_sequence = line[:-1]
            AA_len = len(AA_sequence)
            if AA_len > MAX_SEQ_LEN:
                skip = True
                pass #ignore protein
            else:
                AA_tokens = tokeniser.GetVectoredPolypeptide(AA_sequence, dict_) #turns AA letters into tokens which are vectors with length 20 in the form of [1, 0, 0, 0, 0 ....]
                primary_padded[:AA_len] = AA_tokens #changes the nonpadded values to acctual values
        elif stage == "[EVOLUTIONARY]\n" and skip != True:
            pass

        elif stage == "[MASK]" and skip != True:
            mask_padded[:AA_len] = line #adds masks values to the padded values
            masks = np.append(masks, mask_padded)

        elif stage == "[TERTIARY]\n" and skip != True:
            current.append([float(i) for i in line.split('\t')])


        elif stage == "[ID]\n" and skip != True:
            if current != []: #put the terirary array into the shape of (-1,3,3) or (Seq_len, Backbone atoms, xyz)
                skip = False
                T = np.array(current)
                T = np.transpose(T)
                T = T.reshape((-1, 3, 3))
                tertiary_padded[:AA_len] = T
                tertiarys = np.append(tertiarys, tertiary_padded)
                current = []



    return list(zip(primarys,tertiarys))

def save(data, filename, path="PreData/"):
    with open(path+filename+".txt", 'w+b') as fp:
        pickle.dump(data, fp)

def load(filename, path="PreData/"):
    with open(path+filename+'.txt', 'rb') as file:
        return pickle.load(file)

def preprocess_file(raw_file_queue):
    for file in raw_file_queue:
        data = getRecord(file)
        save(data, file+'_preprocessed')
    print("Preprocessing Done!!!")

def loader(filename, raw_path=''):
    if os.path.isfile('PreData/'+filename+'.txt'):
        print("Found preprocessed data...")
        return load(filename)
    else:
        print("Preprocessing data")
        data = getRecord(raw_path+filename)
        save(data, filename)
        return data

if __name__ == '__main__':
    data = loader('protein_test')
    print(data)