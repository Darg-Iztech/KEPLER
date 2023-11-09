#!/bin/bash

python evaluate_transe_inductive.py --entity_embeddings files_for_inductive_cite/EntityEmb_cite.npy \
		--relation_embeddings files_for_inductive_cite/RelEmb_cite.npy \
		--dim 768 \
		--entity2id files_for_inductive_cite/entity2id.json \
		--relation2id files_for_inductive_cite/relation2id.json \
		--dataset files_for_inductive_cite/citation_triplets_test.txt

