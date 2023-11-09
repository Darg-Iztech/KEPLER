#!/bin/bash
TOTAL_UPDATES=124000    # Total number of training steps
                        # EPOCHS = TOTAL_UPDATES / MAX_SENTENCES
                        # 1 epochs take ? minutes in AUDP

WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
LR=6e-04                # Peak LR for polynomial LR scheduler
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size # Set to 16 in AUDP, Set to 2 in DARG 
NUM_NODES=1		# Number of machines

CHECKPOINT_PATH="checkpoints_nhead_disabled_mlmetke_acl_cite" #Directory to store the checkpoints
UPDATE_FREQ=`expr 784 / $NUM_NODES` # Increase the batch size
DATA_DIR="/projects/KEPLER_DATA"
MLM_DATA=$DATA_DIR/ACL_CITE_MLM/preprocessed
KE_DATA=$DATA_DIR/ACL_CITE_KE/preprocessed
ROBERTA_PATH=$DATA_DIR/roberta_base_model.pt # Path to the original roberta model

DIST_SIZE=`expr $NUM_NODES \* 1`

fairseq-train $MLM_DATA \
        --KEdata $KE_DATA \
        --restore-file $ROBERTA_PATH \
        --save-dir $CHECKPOINT_PATH \
        --max-sentences $MAX_SENTENCES \
        --tokens-per-sample 512 \
        --task MLMetKE \
        --sample-break-mode complete \
        --required-batch-size-multiple 1 \
        --arch roberta_base \
        --criterion MLMetKE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --negative-sample-size 1 \
        --ke-model TransE \
        --init-token 0 \
        --separator-token 2 \
        --gamma 4 \
        --nrelation 1 \
        --skip-invalid-size-inputs-valid-test \
        --fp16 --fp16-init-scale 2 --threshold-loss-scale 1 --fp16-scale-window 128 \
        --reset-optimizer --distributed-world-size ${DIST_SIZE} --ddp-backend no_c10d --distributed-port 23456 \
        --log-format simple --log-interval 1 \
        --no-epoch-checkpoints
