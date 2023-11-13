#!/bin/bash

MLM_DATA=/projects/KEPLER_DATA/ACL_CITE_MLM

# Required directory structure:
# MLM_DATA
# ├── raw_corpus
# │   └── all.txt
# └── gpt2_bpe
#     ├── dict.txt
#     ├── encoder.json
#     └── vocab.bpe


echo "Removing ids from raw corpus and converting the values to numerical..."
python convert.py \
		--data_dir $MLM_DATA \
		--raw_corpus $MLM_DATA/raw_corpus/all.txt \
		--converted_corpus $MLM_DATA/corpus/all.txt

echo "Splitting converted corpus into train/valid/test..."
python split.py \
		--input_file $MLM_DATA/corpus/all.txt \
		--output_dir $MLM_DATA/corpus

echo "BPE Encoding the converted corpus..."
for SPLIT in train valid test; do \
    python ../../roberta/multiprocessing_bpe_encoder.py \
        --encoder-json $MLM_DATA/gpt2_bpe/encoder.json \
        --vocab-bpe $MLM_DATA/gpt2_bpe/vocab.bpe \
        --inputs $MLM_DATA/corpus/$SPLIT.txt \
        --outputs $MLM_DATA/corpus/$SPLIT.bpe \
        --keep-empty \
        --workers 40; \
done

[ -d "$MLM_DATA/preprocessed" ] && rm -rf "$MLM_DATA/preprocessed"

echo "Binarizing..."
python ../../../fairseq_cli/preprocess.py \
    --only-source \
    --srcdict $MLM_DATA/gpt2_bpe/dict.txt \
    --trainpref $MLM_DATA/corpus/train.bpe \
    --validpref $MLM_DATA/corpus/valid.bpe \
    --testpref $MLM_DATA/corpus/test.bpe \
    --destdir $MLM_DATA/preprocessed/ \
    --workers 40
    
# https://github.com/THU-KEG/KEPLER/blob/main/examples/roberta/README.pretraining.md
