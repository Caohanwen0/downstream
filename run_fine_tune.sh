#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=12423
NNODES=1
NODE_RANK=0

GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
BASE_PATH="/data1/private/caohanwen/OpenSoCo"
SAVE_PATH="/data1/private/caohanwen/OpenSoCo/downstream/save"
DATASET_PATH="/data1/private/caohanwen/OpenSoCo/downstream/datasets"
DATASET_NAME="sentiment/semeval2017task4a"
MODEL_NAME="deberta_prenorm_OpenSoCo_en"
MODEL_TRAIN_STEP=112500
CONFIG="deberta_prenorm"

OPTS=""
OPTS+=" --max-length 512"
OPTS+=" --lr 1e-5"
OPTS+=" --epochs 30"
OPTS+=" --warmup-ratio 0.01"
OPTS+=" --batch-size 32"
OPTS+=" --log-iters 10"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset-path ${DATASET_PATH}"
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --save-tensorboard False"
OPTS+=" --model-config config/${CONFIG}.json"
OPTS+=" --load ${BASE_PATH}/downstream/model/en_deberta/checkpoint-${MODEL_TRAIN_STEP}.pt"
OPTS+=" --save ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/${MODEL_NAME}/${MODEL_TRAIN_STEP}"
OPTS+=" --tokenizer tokenizer/en.json"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/downstream/fine_tune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/${MODEL_NAME}/${MODEL_TRAIN_STEP}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/${MODEL_NAME}/${MODEL_TRAIN_STEP}/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
