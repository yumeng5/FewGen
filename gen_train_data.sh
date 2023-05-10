CUDA_ID=$1
export CUDA_VISIBLE_DEVICES=$CUDA_ID
TASK=$2
SEED=$3
LABEL=$4
MODEL_PATH=generator/$TASK/$SEED
NUM_GEN=$5
CORPUS_PATH="None"

TASK=${TASK,,}

case $TASK in
    cola)
        case $LABEL in 
            0)
            EXTRA="--temperature 10"
            ;;
            1)
            EXTRA="--temperature 0.3"
            ;;
        esac
        ;;
    sst-2)
        EXTRA="--temperature 0.5"
        ;;
    mnli)
        CORPUS_PATH="pretrain_corpus/wiki_short.txt"
        ;;
    mrpc)
        CORPUS_PATH="pretrain_corpus/wiki_long.txt"
        ;;
    qnli)
        CORPUS_PATH="pretrain_corpus/openwebtext_questions.txt"
        ;;
    qqp)
        CORPUS_PATH="pretrain_corpus/openwebtext_questions.txt"
        ;;
    rte)
        CORPUS_PATH="pretrain_corpus/wiki_long.txt"
        ;;
esac

python src/gen_train_data.py --model_name_or_path "${MODEL_PATH}" \
        --save_dir "gen_res_${TASK}_${SEED}" \
        --label $LABEL \
        --temperature 0 \
        --p 1.0 \
        --k 10 \
        --num_gen $NUM_GEN \
        --max_len 50 \
        --print_res \
        --pretrain_corpus_dir $CORPUS_PATH \
        --task $TASK \
        $EXTRA
