"""
########################
### preprocessing.py ###
########################

~ Will Bennett 17/06/2021

Turns the proteinnet dataset into usable data for the model to make predictions from and train form.
"""
import torch 

def getRecord(filename):
    file = open(filename, 'r')#gets txt file
    stage = ""
    x = []
    y = []
    current = [] #for stage where multiple lines are used
    for line in file:
        if line[0] == '[':
            stage = line
        elif stage == "[PRIMARY]\n":
            print(line)
            x.append(line[:-2])
        elif stage == "[EVOLUTIONARY]\n":
            pass
        elif stage == "[TERTIARY]\n":
            # print(line)
            current.append([float(i) for i in line.split('\t')])
        elif stage == "[ID]\n":
            if current != []:
                y.append(current)
                current = []
    return x, y

    #print(file.read())


if __name__ == '__main__':
    x,y = getRecord('protein_test')
    print(x)
    print(y)
