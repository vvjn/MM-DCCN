#!/usr/bin/env bash
# Authors: Vipin Vijayan, Thamme Gowda
# Created : Nov 06, 2017

# Usage:
# sh bpe_pipeline_mmod.sh run1 [preprocess|train|translate|evaluate] ../MM-DCCN ../multi30k-wmt18/task1-data/en-de ../multi30k-wmt18/task1-data/features_resnet50 ../multi30k-wmt18/task1-data/features_dccn

# update these variables
NAME=$1
OUT="onmt-runs/$NAME"

ONMT=$3

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
# source ~/.bashrc
# source activate py3

DATA=$4
BPE_SRC=$DATA/codes
BPE_TGT=$DATA/codes
TRAIN_SRC=$DATA/train.lc.norm.tok.bpe10000.en
TRAIN_TGT=$DATA/train.lc.norm.tok.bpe10000.de
VALID_SRC=$DATA/val.lc.norm.tok.bpe10000.en
VALID_TGT=$DATA/val.lc.norm.tok.bpe10000.de
TEST_SRC=$DATA/test_2016_flickr.lc.norm.tok.bpe10000.en
TEST_TGT=$DATA/test_2016_flickr.lc.norm.tok.bpe10000.de

FEATS_DATA=$5
BUTD_DATA=$6

GPUARG="" # default
GPUARG="0"

#====== EXPERIMENT BEGIN ======

function lines_check {
    l1=`cat $1 | wc -l`
    l2=`cat $2 | wc -l`
    if [[ $l1 != $l2 ]]; then
        echo "ERROR: Record counts doesnt match between: $1 and $2"
        exit 2
    fi
}

case $2 in

    preprocess)

        # Check if input exists
        for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
            if [[ ! -f "$f" ]]; then
                echo "Input File $f doesn't exist. Please fix the paths"
                exit 1
            fi
        done
        
        lines_check $TRAIN_SRC $TRAIN_TGT
        lines_check $VALID_SRC $VALID_TGT
        lines_check $TEST_SRC $TEST_TGT
        
        echo "Output dir = $OUT"
        [ -d $OUT ] || mkdir -p $OUT
        [ -d $OUT/data ] || mkdir -p $OUT/data
        [ -d $OUT/models ] || mkdir $OUT/models
        [ -d $OUT/test ] || mkdir -p  $OUT/test
        
        echo "Step 1a: Preprocess inputs"
        cp $BPE_SRC $OUT/data/code.src
        cp $TRAIN_SRC $OUT/data/train.src
        cp $VALID_SRC $OUT/data/valid.src
        cp $TEST_SRC $OUT/data/test.src
        cp $BPE_TGT $OUT/data/code.tgt
        cp $TRAIN_TGT $OUT/data/train.tgt
        cp $VALID_TGT $OUT/data/valid.tgt
        cp $TEST_TGT $OUT/data/test.tgt

        #: <<EOF
        echo "Step 1b: Preprocess"
        python $ONMT/preprocess.py \
            -train_src $OUT/data/train.src \
            -train_tgt $OUT/data/train.tgt \
            -valid_src $OUT/data/valid.src \
            -valid_tgt $OUT/data/valid.tgt \
            -save_data $OUT/data/processed
        
        exit
        ;;


    train)

        echo "Step 2: Train"
        GPU_OPTS=""
        if [[ ! -z $GPUARG ]]; then
            GPU_OPTS="-gpuid $GPUARG"
        fi
        CMD="python $ONMT/train_mmod.py -data $OUT/data/processed -save_model $OUT/models/$NAME $GPU_OPTS -path_to_train_img_feats $FEATS_DATA/train-resnet50-res4frelu.npy  -path_to_train_attr $BUTD_DATA/train-img_attr.npy -path_to_train_img_mask $BUTD_DATA/train-img_mask.npy -path_to_valid_img_feats $FEATS_DATA/val-resnet50-res4frelu.npy  -path_to_valid_attr $BUTD_DATA/val-img_attr.npy -path_to_valid_img_mask $BUTD_DATA/train-img_mask.npy -num_regions 10 -encoder_type transformer -decoder_type transformer --multimodal_model_type dcap -batch_type sents -batch_size 3700 -valid_batch_size 1000 -epochs 25"
        echo "Training command :: $CMD"
        eval "$CMD"
        
        ;;


    translate)
        
        # select a model with high accuracy and low perplexity
        # TODO: currently using linear scale, maybe not be the best
        model=`ls $OUT/models/*.pt| awk -F '_' 'BEGIN{maxv=-1000000} {score=$(NF-3)-$(NF-1); if (score > maxv) {maxv=score; max=$0}}  END{ print max}'`
        echo "Chosen Model = $model"
        if [[ -z "$model" ]]; then
            echo "Model not found. Looked in $OUT/models/"
            exit 1
        fi
        
        GPU_OPTS=""
        if [ ! -z $GPUARG ]; then
            GPU_OPTS="-gpu $GPUARG"
        fi
        
        echo "Step 3a: Translate Test"
        python $ONMT/translate_mmod.py -model $model \
               -src $OUT/data/test.src \
               -path_to_test_img_feats $FEATS_DATA/test-img_feats.npy \
               -path_to_test_attr $BUTD_DATA/test-img_attr.npy \
               -path_to_test_img_mask $BUTD_DATA/test-img_mask.npy \
               --multimodal_model_type dcap \
               -output $OUT/test/test.out \
               -replace_unk  -verbose $GPU_OPTS > $OUT/test/test.log
        
        echo "Step 3b: Translate Dev"
        python $ONMT/translate_mmod.py -model $model \
            -src $OUT/data/valid.src \
            -path_to_test_img_feats $FEATS_DATA/val-img_feats.npy \
            -path_to_test_attr $BUTD_DATA/val-img_attr.npy \
            -path_to_test_img_mask $BUTD_DATA/val-img_mask.npy \
            --multimodal_model_type dcap \
            -output $OUT/test/valid.out \
            -replace_unk -verbose $GPU_OPTS > $OUT/test/valid.log
        
        if [[ "$BPE" == *"tgt"* ]]; then
            echo "BPE decoding/detokenising target to match with references"
            mv $OUT/test/test.out{,.bpe}
            mv $OUT/test/valid.out{,.bpe} 
            cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
            cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out
        fi

        ;;


    evaluate)

        echo "Step 4a: Evaluate Test"
        perl $ONMT/tools/multi-bleu-detok.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
        perl $ONMT/tools/multi-bleu-detok.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu
        
        echo "Step 4b: Evaluate Dev"
        perl $ONMT/tools/multi-bleu-detok.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
        perl $ONMT/tools/multi-bleu-detok.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

        ;;

    all)
        bash "${BASH_SOURCE[0]}" $1 preprocess
        bash "${BASH_SOURCE[0]}" $1 train
        bash "${BASH_SOURCE[0]}" $1 translate
        bash "${BASH_SOURCE[0]}" $1 evaluate
        ;;

esac

        #===== EXPERIMENT END ======
         
