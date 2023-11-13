#!/bin/bash

KE_DATA=/projects/KEPLER_DATA/ACL_CITE_KE

# Required directory structure:
# KE_DATA
# ├── raw_corpus
# │   └── all.txt
# ├── gpt2_bpe
# │   ├── dict.txt
# │   ├── encoder.json
# │   └── vocab.bpe
# ├── id_mapping
# │   └── citations_per_paper.json
# └── triplets
#     └── all.txt

echo "Splitting triplets into train/valid/test..."
python split.py \
		--input_file $KE_DATA/triplets/all.txt \
		--output_dir $KE_DATA/triplets

echo "Removing ids from raw corpus and converting both triplets and corpus values to numerical..."
python convert.py \
		--data_dir $KE_DATA \
		--raw_corpus $KE_DATA/raw_corpus/all.txt \
		--train_triplets $KE_DATA/triplets/train.txt \
		--valid_triplets $KE_DATA/triplets/valid.txt \
		--test_triplets $KE_DATA/triplets/test.txt \
		--converted_corpus $KE_DATA/corpus/all.txt \
		--converted_train_triplets $KE_DATA/triplets/train.bpe \
		--converted_valid_triplets $KE_DATA/triplets/valid.bpe \
		--converted_test_triplets $KE_DATA/triplets/test.bpe

echo "Converting citation_per_paper.json to numerical using entity2id.json..."
python convert_json.py --data_dir $KE_DATA

echo "BPE Encoding the converted corpus..."
#mkdir -p gpt2_bpe
#wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
#wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
python ../../roberta/multiprocessing_bpe_encoder.py \
		--encoder-json $KE_DATA/gpt2_bpe/encoder.json \
		--vocab-bpe $KE_DATA/gpt2_bpe/vocab.bpe \
		--inputs $KE_DATA/corpus/all.txt \
		--outputs $KE_DATA/corpus/all.bpe \
		--keep-empty \
		--workers 40


[ -d "$KE_DATA/preprocessed" ] && rm -rf "$KE_DATA/preprocessed"

echo "Negative Sampling..."
python KGpreprocessAdvanced.py \
		--negative_sampling_size 1 \
		--dumpPath $KE_DATA/preprocessed \
		--ent_desc $KE_DATA/corpus/all.bpe \
		--train $KE_DATA/triplets/train.bpe \
		--valid $KE_DATA/triplets/valid.bpe \
		--test $KE_DATA/triplets/test.bpe \
		--json $KE_DATA/id_mapping/citations_per_paper_id.json \
		--negative_sampling_type local

echo "Binarizing..."
for SPLIT in head negHead tail negTail; do \
	python ../../../fairseq_cli/preprocess.py \
	    --only-source \
	    --srcdict $KE_DATA/gpt2_bpe/dict.txt \
	    --trainpref $KE_DATA/preprocessed/$SPLIT/train.bpe \
	    --validpref $KE_DATA/preprocessed/$SPLIT/valid.bpe \
	    --testpref $KE_DATA/preprocessed/$SPLIT/test.bpe \
	    --destdir $KE_DATA/preprocessed/$SPLIT \
	    --workers 40; \
done
