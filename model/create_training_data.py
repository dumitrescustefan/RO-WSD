import logging, os, sys, json, torch, pickle
import torch.nn as nn
from rowordnet import Synset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import rowordnet as rwn
from tqdm.autonotebook import tqdm as tqdm

with open("../dataset.pickle","rb") as f:
    dataset = pickle.load(f)

train, dev, test = [], [], []
import random

print(f"total literals: {len(dataset)}")
for e in dataset: #for each literal
    d_list = dataset[e]
    for entry in d_list: # for each sentence in the literal
        r = random.random()
        #print(entry)
        instance = {
            "prefix_text": entry["text_prefix"],
            "text": entry["text"],
            "suffix_text": entry["text_postfix"],
            "choices": entry["synsets"].split(),
            "target": entry["correct_synset_id"]
        }

        if r<0.7:
            train.append(instance)
        elif r<0.8:
            dev.append(instance)
        else:
            test.append(instance)

print(f"TRAIN len: {len(train)}")
print(f"DEV len: {len(dev)}")
print(f"TEST len: {len(test)}")

with open("train.json", "w", encoding="utf8") as f:
    json.dump(train, f,indent=2, ensure_ascii=False)
with open("dev.json", "w", encoding="utf8") as f:
    json.dump(dev, f, indent=2, ensure_ascii=False)
with open("test.json", "w", encoding="utf8") as f:
    json.dump(test, f, indent=2, ensure_ascii=False)

"""with open("export.csv","w",encoding="utf8") as f:
    users = set()
    for literal in dataset:
        for obj in dataset[literal]:
            users.add(obj["user_id"])

    users = sorted(list(users))
    f.write("Nume, numar propozitii\n")
    for user in users:
        f.write(f"{user}, {_get_sentences_done_for_this_user(user, dataset)}\n")
"""

sys.exit(0)


wn = rwn.RoWordNet()

import os

#with open("synset2sentence-oscar.pickle", "rb") as f:
with open("synset2sentence.pickle", "rb") as f:
    literal_synset2sentences = pickle.load(f)

#with open("literal2synset_sentence_counts-oscar.pickle", "rb") as f:
with open("literal2synset_sentence_counts.pickle", "rb") as f:
    literal2synset_sentence_counts = pickle.load(f)

for literal in literal2synset_sentence_counts:
    print(literal2synset_sentence_counts[literal])

print("_"*100)
for literal in literal2synset_sentence_counts:
    greater_than_zero = 0
    for synset_id in literal2synset_sentence_counts[literal]:
        if literal2synset_sentence_counts[literal][synset_id] > 0:
            greater_than_zero += 1
    if greater_than_zero > 1:
        print(literal2synset_sentence_counts[literal])

kk = {'ENG30-08426111-n': 10, 'ENG30-07137622-n': 12, 'ENG30-13592598-n': 42, 'ENG30-13592384-n': 32, 'ENG30-00368302-n': 36, 'ENG30-11439031-n': 122, 'ENG30-07312221-n': 22, 'ENG30-14009763-n': 7}
for id in kk:
    w, sent = literal_synset2sentences[id][0]
    print()
    print(sent)
    print(w)
    print()

print(len(literal2synset_sentence_counts))

import statistics

cnt = 0
avg = []
t_literal2stdev = {}
for literal in literal2synset_sentence_counts:
    vv = []
    for sid, val in literal2synset_sentence_counts[literal].items():
        vv.append(val)
    t_literal2stdev[literal] = statistics.stdev(vv)
    c = len(literal2synset_sentence_counts[literal])
    cnt += c
    avg.append(c)
print(f"total literal---synset_ids {cnt}, average{sum(avg) / len(avg)}")

# sorted
ddd = {k: v for k, v in sorted(t_literal2stdev.items(), key=lambda item: item[1])}
desc_ordered_literals = []
for k, v in ddd.items():
    print(f"{k}:{v}")
    desc_ordered_literals.append(k)

desc_ordered_literals = desc_ordered_literals[::-1]
print(desc_ordered_literals)


final = {}
total_sentences = 0
no_sentences_for_this_literal_and_synset = 0
import random

for literal in desc_ordered_literals:
    print("\n"+literal)
    print(literal2synset_sentence_counts[literal])

    d = {}
    sc = 0
    for synset_id in literal2synset_sentence_counts[literal]:
        #print(literal2synset_sentence_counts[literal])
        if synset_id not in literal_synset2sentences:
            no_sentences_for_this_literal_and_synset += 1
            continue
        sentences = literal_synset2sentences[synset_id]
        d[synset_id] = random.sample(sentences, min(len(sentences),3))

        sc += len(d[synset_id])
    if sc > 3:
        final[literal] = d
        total_sentences += len(d[synset_id])

print("total sentences")
print(total_sentences)
print(f"no_sentences_for_this_literal_and_synset : {no_sentences_for_this_literal_and_synset}")


"""
literal2synset : cheie literal (lema) -> synset_id, count propozitii
synset2sentence: synset_id -> lista de propozitii cu cuvantul in sine 

how many literals that are ambiguous -> nr de chei in literal2synset: 8K cuvinte ambigue


un literal poate sa aiba 5 synseturi
un synset are 3 literali

imbalans = pt fiecare literal, serie cu stdev maxim 
sortat descrescator, luat minim 3 prop per synset
exclus literali care au synsets cu 0 intrari?????

luat 3 propozitii de fiecare 
"""




