#!/bin/bash

TEST_DATA=/projects/KEPLER_DATA/ACL_CITE_TEST
CKPT_DIR=/projects/KEPLER/checkpoints_nhead_disabled_mlmetke_acl_cite_local

# Run ../Pretrain/dataprep_test.sh to generate the required directory structure

# Required directory structure: 
# TEST_DATA
# ├── raw_corpus
# │   └── all.txt   --> This should include all entity ids in triplets/test.txt
# ├── corpus
# │   ├── all.txt   --> This should include all entity ids in triplets/test.txt
# │   └── all.bpe   --> This should include all entity ids in triplets/test.txt
# ├── gpt2_bpe
# │   ├── dict.txt
# │   ├── encoder.json
# │   └── vocab.bpe
# ├── id_mapping
# │   ├── citations_per_paper.json
# │   ├── relation2id.json
# │   └── entity2id.json
# ├── embeddings
# └── triplets
#     ├── test.txt
#     └── test.bpe

echo "Creating embeddings directory..."
if [ ! -d "$TEST_DATA/embeddings" ]; then
        mkdir "$TEST_DATA/embeddings"
else
        rm -r "$TEST_DATA/embeddings"
        mkdir "$TEST_DATA/embeddings"
fi

echo "Copying dict.txt to checkpoint directory..."
if [ ! -e "$CKPT_DIR/dict.txt" ]; then
        cp $TEST_DATA/gpt2_bpe/dict.txt $CKPT_DIR
fi

echo "Generating embeddings..."
python generate_embeddings.py \
        --data $TEST_DATA/corpus/all.bpe \
        --ckpt_dir $CKPT_DIR \
        --ckpt checkpoint_best.pt \
        --dict $TEST_DATA/gpt2_bpe/dict.txt \
        --ent_emb $TEST_DATA/embeddings/EntityEmb.npy \
        --rel_emb $TEST_DATA/embeddings/RelEmb.npy \
        --batch_size 64

# echo "Evaluating..."
python evaluate_transe_inductive.py \
        --entity_embeddings $TEST_DATA/embeddings/EntityEmb.npy \
        --relation_embeddings $TEST_DATA/embeddings/RelEmb.npy \
        --dim 768 \
        --entity2id $TEST_DATA/id_mapping/entity2id.json \
        --relation2id $TEST_DATA/id_mapping/relation2id.json \
        --dataset $TEST_DATA/triplets/test.txt
