CUDA_ID=$1
export CUDA_VISIBLE_DEVICES=$CUDA_ID

# Arguments
TASK=$2
MODEL=ctrl
SEED=$3
BSZ=2
LR=5e-3

# Meta weighting parameters
WEIGHT_LR=1e-2
META_LR=2e-2
EPOCHS=20

# Save and evaluation parameters
SAVE=1
EVAL=100

TRAIN_MODE=prefix-infix

if (( $EVAL > 0 ))
then
    EVAL_EXTRA="--evaluation_strategy steps --eval_steps $EVAL"
else
    EVAL_EXTRA=""
fi

if (( $SAVE > 0 ))
then
    SAVE_EXTRA=""
else
    SAVE_EXTRA="--no_save"
fi

echo "Training ${MODEL}, with seed ${SEED} for ${TASK}"
python train_generator.py \
    --task_name $TASK \
    --data_dir data/k-shot/$TASK/16-$SEED \
    --model_name_or_path $MODEL \
    --max_seq_length 128 \
    --first_sent_limit 100 \
    --do_train \
    --do_eval \
    --logging_steps 5 \
    --weight_net_lr $WEIGHT_LR \
    --meta_lr $META_LR \
    --learning_rate $LR \
    --per_device_train_batch_size $BSZ \
    --train_mode $TRAIN_MODE \
    --num_train_epochs $EPOCHS \
    --meta_weight \
    --output_dir generator/${TASK}/${SEED}/ \
    --overwrite_output_dir \
    $SAVE_EXTRA $EVAL_EXTRA
