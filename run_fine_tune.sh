#! /bin/bash
MASTER_ADDR=localhost
# MASTER_PORT=12423
NNODES=1
NODE_RANK=0

GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
BASE_PATH="/data1/private/caohanwen/OpenSoCo"
SAVE_PATH="/data1/private/caohanwen/OpenSoCo/downstream/save"
DATASET_PATH="/data1/private/caohanwen/OpenSoCo/downstream/datasets"
LOAD="model/en_deberta/checkpoint-357500.pt"

export DATASET_NAME
export LR
export RANDOM_SEED

OPTS=""
OPTS+=" --load ${LOAD}"
OPTS+=" --tokenizer tokenizer/en.json"
OPTS+=" --max-length 512"
OPTS+=" --lr ${LR}"
OPTS+=" --seed ${RANDOM_SEED}"
OPTS+=" --epochs 30"
OPTS+=" --warmup-ratio 0.01"
OPTS+=" --batch-size 32"
OPTS+=" --gradient-accumulate 1"
OPTS+=" --log-iters 20"
OPTS+=" --model-config model/en_deberta/deberta_prenorm.json"
# OPTS+=" --checkpoint ${CHECKPOINT}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset-path ${DATASET_PATH}"
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --save ${SAVE_PATH}/${DATASET_NAME}/checkpoint=${CHECKPOINT}/lr=${LR}/seed=${RANDOM_SEED}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/downstream/fine_tune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}/${DATASET_NAME}/${CHECKPOINT}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${DATASET_NAME}/${CHECKPOINT}/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
