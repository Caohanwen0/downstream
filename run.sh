#! /bin/bash

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=12423
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_IP}"
    MASTER_PORT="${MASTER_PORT}"
    NNODES=1
    NODE_RANK="$MARSV2_RANK"
fi

GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
BASE_PATH="/weka-jd/prod/public/permanent/group_liuzhiyuan/chenhuimin/workspaces/bm_train_codes"
SAVE_PATH="/weka-jd/prod/public/permanent/group_liuzhiyuan/chenhuimin/workspaces/bm_train_codes/downstream/save"
DATASET_PATH="/weka-jd/prod/public/permanent/group_liuzhiyuan/chenhuimin/workspaces/bm_train_codes/downstream/datasets"
DATASET_NAME="sentiment/semeval2017task4a"
MODEL_NAME="roberta-base_OpenSoCo_en"
MODEL_TRAIN_STEP=160500
CONFIG="roberta-base_prenorm"

OPTS=""
OPTS+=" --lr 1e-5"
OPTS+=" --epochs 32"
OPTS+=" --dataset-path ${DATASET_PATH}"
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --model-config config/${CONFIG}.json"
OPTS+=" --load ${BASE_PATH}/${MODEL_NAME}/1e-4-init-embed/checkpoints/checkpoint-${MODEL_TRAIN_STEP}.pt"
OPTS+=" --save ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/${MODEL_NAME}/${MODEL_TRAIN_STEP}"
OPTS+=" --tokenizer tokenizer/en.json"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} delta_tune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/${MODEL_NAME}/${MODEL_TRAIN_STEP}

if [[ $NODE_RANK == 0 ]]&&[[ $DLS_TASK_NUMBER == 1 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/${MODEL_NAME}/${MODEL_TRAIN_STEP}/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
