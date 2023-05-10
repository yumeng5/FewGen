CUDA_ID=$1
export CUDA_VISIBLE_DEVICES=$CUDA_ID

TASK=$2
MODEL=roberta-large
SM=0.15
SEED=$3
METHOD=prompt
LR=$4
BS=$5
OUT_DIR=fewshot_result/$TASK/$SEED/LR-$LR-BS-$BS

TASK_EXTRA=""
case $TASK in
    CoLA)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        METHOD=finetune
        ;;
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
esac

python train_classifier.py \
    --task_name $TASK \
    --data_dir data/k-shot/$TASK/16-$SEED \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --smooth $SM \
    --eval_steps 100 \
    --evaluate_during_training \
    --model_name_or_path $MODEL \
    --finetune_type $METHOD \
    --max_seq_length 128 \
    --first_sent_limit 100 \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps 1 \
    --learning_rate $LR \
    --max_steps 1000 \
    --output_dir $OUT_DIR \
    --seed $SEED \
    --template $TEMPLATE \
    --mapping $MAPPING \
    --logging_dir $OUT_DIR \
    --logging_steps 20 \
    --warmup_ratio 0.1 \
    $TASK_EXTRA

