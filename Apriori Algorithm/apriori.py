import sys
import pandas as pd
import numpy as np
import re

c1=[0] * 100
c2=[[0 for col in range(100)] for row in range(100)]
candidate=[]
n=0

filename = sys.argv[1]
minsupp = float(sys.argv[2])
minconf = float(sys.argv[3])

file = open(filename,'r')
data = file.readline()
while data:
    n=n+1
    numbers = re.findall("\d+",data)
    for i in range(1,len(numbers)):
        c1[int(numbers[i])] = c1[int(numbers[i])] + 1
    data = file.readline()

file.close();

candi=[0]
for i in range(0,100):
    if(c1[i]/n >= minsupp):    #Step 1: Find frequent 1-itemsets
         candi.append(i)
candi.pop(0)

#Step 2: Generate candidate 2-itemsets
for i in range(len(candi)-1):
    for j in range(i+1,(len(candi))):
        can=[]
        can.append(candi[i])
        can.append(candi[j])
        candidate.append(can)

file = open(filename,'r')
data = file.readline()
while data:
    numbers = re.findall("\d+",data)
    numbers.pop(0)
    for i in range(len(candidate)):
        if (str(candidate[i][0]) in numbers) and (str(candidate[i][1]) in numbers):
            c2[candidate[i][0]][candidate[i][1]] =  c2[candidate[i][0]][candidate[i][1]] + 1
    data = file.readline()

file.close();


#Step 3: Find frequent 2-itemsets
length = len(candidate)
candidate2 = []
for i in range(length):
    support = c2[candidate[i][0]][candidate[i][1]] / n
    if(support >= minsupp ):
        can2 = []
        can2.append(candidate[i][0])
        can2.append(candidate[i][1])
        candidate2.append(can2)


print("Association rules found:")
#Sept 4: Generate association rules
for i in range (len(candidate2)):
    support = c2[candidate2[i][0]][candidate2[i][1]]/n
    confidence1 = c2[candidate2[i][0]][candidate2[i][1]]/c1[candidate2[i][0]]
    confidence2= c2[candidate2[i][0]][candidate2[i][1]]/c1[candidate2[i][1]]
    if (confidence1 >= minconf):
        print( candidate2[i][0],"- >" , candidate2[i][1], "(support = ", support,", confidence = " ,confidence1)
    if(confidence2 >= minconf):
        print( candidate2[i][1],"- >", candidate2[i][0],"(support = " , support ,", confidence = ", confidence2)
