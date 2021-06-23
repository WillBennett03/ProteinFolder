"""
####################
### tokeniser.py ###
####################

~ ill Bennett 09/06/2021

This modules allows for converting a string of Amino acids into an array of vectors 
similar to NLP word embedding.

"""
import pandas as pd
import numpy as np
import torch
import json

def get_aminoacids(Filename="AminoAcids.json", path="JSON/"): #gets the features of each amino acid
    dict_ = json.load(open(path+Filename))
    df =  pd.DataFrame.from_dict(dict_, orient='index')
    df["pK"] /= 1000 # Scaling it to a decimal
    # print(df.head())
    # print(len(df.index))
    return df, dict_

def give_vector(AA, Dict):
    index = list(Dict.keys()).index(AA)
    temp = torch.zeros(20).tolist()
    temp[index] = 1
    # residue = list(Dict[AA].values())
    # temp.extend(residue)
    return temp

def GetVectoredPolypeptide(polypeptide, dict):
    x = []
    for AA in polypeptide:
        x.append(give_vector(AA, dict))
    return torch.tensor(x, dtype=torch.int64)

if __name__ =='__main__': #test
    df, dict = get_aminoacids()
    p = 'ARN'
    tensor = GetVectoredPolypeptide(p, dict)
    print(tensor)