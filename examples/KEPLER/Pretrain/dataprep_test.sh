#!/bin/bash

TEST_DATA=/projects/KEPLER_DATA/ACL_CITE_TEST

# Required directory structure:
# TEST_DATA
# ├── raw_corpus
# │   └── all.txt   --> This should include all entity ids in triplets/test.txt
# ├── gpt2_bpe
# │   ├── dict.txt
# │   ├── encoder.json
# │   └── vocab.bpe
# ├── id_mapping
# │   └── citations_per_paper.json
# ├── embeddings
# └── triplets
#     └── test.txt

echo "Removing ids from raw corpus and converting both triplets and corpus values to numerical..."
python convert.py \
		--data_dir $TEST_DATA \
		--raw_corpus $TEST_DATA/raw_corpus/all.txt \
        --test_triplets $TEST_DATA/triplets/test.txt \
		--converted_corpus $TEST_DATA/corpus/all.txt \
		--converted_test_triplets $TEST_DATA/triplets/test.bpe

echo "Converting citation_per_paper.json to numerical using entity2id.json..."
python convert_json.py --data_dir $TEST_DATA

echo "BPE Encoding the converted corpus..."
#mkdir -p gpt2_bpe
#wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
#wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
python ../../roberta/multiprocessing_bpe_encoder.py \
    --encoder-json $TEST_DATA/gpt2_bpe/encoder.json \
    --vocab-bpe $TEST_DATA/gpt2_bpe/vocab.bpe \
    --inputs $TEST_DATA/corpus/all.txt \
    --outputs $TEST_DATA/corpus/all.bpe \
    --keep-empty \
    --workers 40;
