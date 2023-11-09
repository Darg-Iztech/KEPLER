import argparse
import sys
sys.path.append("/projects/KEPLER/graphvite/python/")

import graphvite as gv
import graphvite.application as gap
import numpy as np
import json
import pickle
from tqdm import tqdm

def main():
    print(gv.__file__)
    print(gap.__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_embeddings', help='numpy of entity embeddings')
    parser.add_argument('--relation_embeddings', help='numpy of relation embeddings')
    parser.add_argument('--entity2id', help='entity name to numpy id json')
    parser.add_argument('--relation2id', help='entity name to numpy id json')
    parser.add_argument('--dim', type=int, help='size of embedding')

    parser.add_argument('--dataset', help="test dataset")
    args = parser.parse_args()
    
    # Building the graph 
    app = gap.KnowledgeGraphApplication(dim=args.dim)
    app.load(file_name=args.dataset)
    app.build()
    app.train(model='TransE', num_epoch=0)

    gv_entity2id = app.graph.entity2id
    gv_relation2id = app.graph.relation2id

    # Load embeddings (Only load the embeddings that appear in the entity2id file)
    entity_embeddings_full = np.load(args.entity_embeddings)
    relation_embeddings_full = np.load(args.relation_embeddings)
    entity2id_ori = json.load(open(args.entity2id))
    relation2id_ori = json.load(open(args.relation2id))

    entity_embeddings = np.zeros((len(gv_entity2id), args.dim), dtype=np.float32) 
    entity2id = {}
    i = 0
    for key in tqdm(gv_entity2id):
        entity2id[key] = i
        entity_embeddings[i] = entity_embeddings_full[entity2id_ori[key]]
        i += 1

    relation_embeddings = np.zeros((len(gv_relation2id), args.dim), dtype=np.float32) 
    relation2id = {}
    i = 0
    for key in tqdm(gv_relation2id):
        relation2id[key] = i
        relation_embeddings[i] = relation_embeddings_full[relation2id_ori[key]]
        i += 1
    
    # Load embeddings to graphvite
    print('load data ......')
    assert(len(relation_embeddings) == len(app.solver.relation_embeddings))
    assert(len(entity_embeddings) == len(app.solver.entity_embeddings))
    app.solver.relation_embeddings[:] = relation_embeddings
    print('loaded relation embeddings')
    app.solver.entity_embeddings[:] = entity_embeddings
    print('loaded entity embeddings')
    
    # (Modified gv) Replace mapping with our own
    app.entity2id = entity2id
    app.relation2id = relation2id
    
    print('start evaluation ......')
    print(app.evaluate('entity_prediction', target="tail", k=15, backend="torch", file_name=args.dataset))#, filter_files=[args.dataset])

def get_mrr_score():
    with open('eval_result.pkl', 'rb') as file:
        predictions = pickle.load(file)

    with open('files_for_inductive_cite/citation_triplets_test.txt', "r") as file:
        ground_truths = [line.rstrip().split("\t")[2] for line in file]

    print(f'Ground:{len(ground_truths)} - Predictions:{len(predictions)} ')

    mrr_score = 0.0
    for i, prediction in enumerate(predictions):
        prediction = [p for p in prediction if p[0][0] == 'T']
        for k, prediction_rank in enumerate(prediction):
            if ground_truths[i] == prediction_rank[0]:
                print("{:>0} {:>8} {:>8} {:<8}".format(i+1, ground_truths[i], k+1, prediction_rank[0]))
                mrr_score += 1/(k+1)
    mrr_score = mrr_score/len(predictions)
    print('\nMRR Score:', mrr_score)

if __name__ == '__main__':
    get_mrr_score()
