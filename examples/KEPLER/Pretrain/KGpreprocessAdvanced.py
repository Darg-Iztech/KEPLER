# coding=utf-8
"""do negative sampling and dump training data"""
import os
import sys
import json
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

parser=argparse.ArgumentParser()
parser.add_argument("--dumpPath", type=str, help="path to store output files, do NOT create it previously")
parser.add_argument("-nss", "--negative_sampling_size", type=int, default=1)
parser.add_argument("--train", type=str, help="file name of training triplets")
parser.add_argument("--valid", type=str, help="file name of validation triplets")
parser.add_argument("--test", type=str, help="file name of test triplets")
parser.add_argument("--ent_desc", type=str, help="path to the entity description file (after BPE encoding)")
parser.add_argument("--json", type=str, help="if set to a json file, use local negative sampling")
parser.add_argument("--negative_sampling_type", type=str, default="local", help="local or global negative sampling")

def getTriplets(path):
    res=[]
    with open(path, "r") as fin:
        lines=fin.readlines()
        for l in lines:
            tmp=[int(x) for x in l.split()]
            res.append((tmp[0], tmp[1], tmp[2]))
    return res

def countFrequency(triplets, start=4):
    count = {}
    for head, relation, tail in triplets:
        hr=",".join([str(head), str(relation)])
        tr=",".join([str(tail), str(-relation-1)])
        if hr not in count:
            count[hr] = start
        else:
            count[hr] += 1
        if tr not in count:
            count[tr] = start
        else:
            count[tr] += 1
    return count
    
def getTrueHeadsAndTails(triplets):
    true_heads = {}
    true_tails = {}

    for head, relation, tail in triplets:
        if (head, relation) not in true_tails:
            true_tails[(head, relation)] = []
        true_tails[(head, relation)].append(tail)
        if (relation, tail) not in true_heads:
            true_heads[(relation, tail)] = []
        true_heads[(relation, tail)].append(head)

    for relation, tail in true_heads:
        true_heads[(relation, tail)] = np.array(list(set(true_heads[(relation, tail)])))
    for head, relation in true_tails:
        true_tails[(head, relation)] = np.array(list(set(true_tails[(head, relation)])))                 
    return true_heads, true_tails

def getTokenCount(s):
    return min(len(s.split()), 512)

def getHeadsOrTails(json_data, selection=(None, None), negative_sampling_type="local"):
    # selection is a tuple of (head, tail)
    if selection == (None, None):
        raise ValueError("Either head or tail should be specified")
    if selection[0] is not None and selection[1] is not None:
        raise ValueError("Only one of head or tail should be specified")

    source = "citations" if selection[0] is None else "paper_contexts" # source is the one that is specified in selection
    target = "paper_contexts" if selection[0] is None else "citations" # target is the one that is not specified in selection

    entities = []
    if negative_sampling_type == "global":
        for entry in json_data:
            t = entry[target]
            entities.extend(t)
    elif negative_sampling_type == "local":
        for entry in json_data:
            # Get only the local heads or tails
            s = selection[0] if selection[0] is not None else selection[1]
            if s in entry[source]:
                t = entry[target]
                entities.extend(t)
    return np.array(list(set(entities)))

def genSample(triplets, args, split, Qdesc, true_heads, true_tails, json_data):
    fHead = open(os.path.join(args.dumpPath, "head", split)+".bpe", "w")
    fTail = open(os.path.join(args.dumpPath, "tail", split)+".bpe", "w")
    fnHead = open(os.path.join(args.dumpPath, "negHead", split)+".bpe", "w")
    fnTail = open(os.path.join(args.dumpPath, "negTail", split)+".bpe", "w")
    rel = []
    nss = args.negative_sampling_size
    nst = args.negative_sampling_type
    num_tokens_list = []
    num_entities=len(Qdesc)
    num_successful_n_sampling = [0, 0] # increment first element by 1 if nHead sampling succeeds, second element for nTail
    num_missing_n_sampling = [0, 0] # increment first element by 1 if nHead sampling fails, second element for nTail

    for h,r,t in tqdm(triplets, unit="triplet"):
        num_tokens = 0
        n_samples = {"nHead": [], "nTail": []}
        for m, mode in enumerate(["nHead", "nTail"]):
            selection = (None, t) if mode == "nHead" else (h, None)
            all_headsOrTails = getHeadsOrTails(json_data, selection, negative_sampling_type=nst)
            true_headsOrTails = true_heads[(r, t)] if mode == "nHead" else true_tails[(h, r)]
            mask = np.in1d(all_headsOrTails, true_headsOrTails, assume_unique=True, invert=True)
            all_nHeadsOrTails = all_headsOrTails[mask]
            if len(all_nHeadsOrTails) < nss:
                num_missing_n_sampling[m] += 1
            else:
                n_samples[mode] = np.random.choice(all_nHeadsOrTails, nss, replace=False)
                num_successful_n_sampling[m] += 1

        # If both nHead and nTail sampling succeeds
        if len(n_samples["nHead"]) > 0 and len(n_samples["nTail"]) > 0: 
            num_tokens += getTokenCount(Qdesc[h])
            num_tokens += getTokenCount(Qdesc[t])
            fHead.write(Qdesc[h])
            fTail.write(Qdesc[t])
            rel.append(r)
            # Write negative samples to file
            for tok in n_samples["nHead"]:
                x = int(tok)
                fnHead.write(Qdesc[x])
                num_tokens += getTokenCount(Qdesc[x])
            for tok in n_samples["nTail"]:
                x = int(tok)
                fnTail.write(Qdesc[x])
                num_tokens += getTokenCount(Qdesc[x])
            num_tokens_list.append(num_tokens)

    fHead.close()
    fTail.close()
    fnHead.close()
    fnTail.close()
    np.save(os.path.join(args.dumpPath, "relation", split)+".npy", np.array(rel))
    np.save(os.path.join(args.dumpPath, "sizes", split)+".npy", np.array(num_tokens_list))

    print(f"> {nst} nHead sampling = {num_successful_n_sampling[0]}")
    print(f"> missing nHead sampling = {num_missing_n_sampling[0]}")
    print(f"> {nst} nTail sampling = {num_successful_n_sampling[1]}")
    print(f"> missing nTail sampling = {num_missing_n_sampling[1]}")


if __name__=='__main__':
    args=parser.parse_args()
    TrainTriplets = getTriplets(args.train)
    ValidTriplets = getTriplets(args.valid)
    TestTriplets = getTriplets(args.test)
    AllTriplets = TrainTriplets + ValidTriplets + TestTriplets
    Qdesc=[]
    with open(args.ent_desc, "r") as fin:
        Qdesc=fin.readlines()
    print(str(datetime.now())+" load finish")
    count = countFrequency(AllTriplets)
    true_heads, true_tails = getTrueHeadsAndTails(AllTriplets)
    os.mkdir(args.dumpPath)
    json.dump(count, open(os.path.join(args.dumpPath, "count.json"), "w"))
    for nm in ["head","tail","negHead","negTail","relation","sizes"]:
        os.mkdir(os.path.join(args.dumpPath, nm))
    print(str(datetime.now()) + " preparation finished")
    json_data = None
    if args.json is not None:
        with open(os.path.join(args.json), "r") as file:
            json_data = json.load(file)
    genSample(TrainTriplets, args, "train", Qdesc, true_heads, true_tails, json_data)
    print(str(datetime.now())+" training set finished")
    genSample(ValidTriplets, args, "valid", Qdesc, true_heads, true_tails, json_data)
    print(str(datetime.now())+" validation set finished")
    genSample(TestTriplets, args, "test", Qdesc, true_heads, true_tails, json_data)
    print(str(datetime.now())+" test set finished")
