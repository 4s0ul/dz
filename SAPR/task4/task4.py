#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import defaultdict
from queue import SimpleQueue
import math


# In[2]:


def find_r1_and_r2(l: dict, A: np.array) -> dict:
    for row in A:
        l[row[0]][0] += 1
        l[row[1]][1] += 1
    return l


# In[3]:



def find_r1to4(l: dict, A: np.array) -> dict:
    for row in A:
        main = row[0]
        sub = row[1]
        l[main][0] += 1
        l[sub][1] += 1
        for subrow in A:
            if subrow[0] == sub:
                l[main][2] += 1
                l[subrow[1]][3] += 1
    return l


# In[4]:


def find_r5(d: dict, A: np.array) -> dict:
    q = SimpleQueue()
    q.put(1)

    r5 = {}

    while not q.empty():
        main = q.get()
        l = []
        for row in A:
            if row[0] == main:
                l.append(row[1])
                q.put(row[1])
        if len(l) > 1:
            for elem in l:
                d[elem][4] += l.__len__() - 1

    return d


# In[5]:


def entropy_calc(graph: np.array) -> float:
    def set_def():
        return [0, 0, 0, 0, 0]

    l = defaultdict(set_def)

    find_r1to4(l, graph)
    find_r5(l, graph)

    l = pd.DataFrame(l) 
    l = l.to_numpy().T 
    n = len(l) 

    s = 0.0 # сумма
    for elem in l:
        for cond in elem:
            if cond > 0:
                p = cond / (n - 1)
                logp = math.log10(p)
                s += p * logp

    return -s


# In[6]:


def pipeline(file):
    A = pd.read_csv(file).to_numpy()
    entropy = entropy_calc(A)
    print(f"Ответ: энтропия равна {entropy:.4f} \n")


# In[7]:


file = "task4.csv"
pipeline(file)


# In[ ]:
