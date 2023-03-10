#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=12432
NNODES=1
NODE_RANK=0

GPUS_PER_NODE=4
CUDA_VISIBLE_DEVICES="1,2,3,4"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
BASE_PATH="/data1/private/caohanwen/OpenSoCo"
SAVE_PATH="/data1/private/caohanwen/OpenSoCo/downstream/save"
DATASET_PATH="/data1/private/caohanwen/OpenSoCo/downstream/datasets"
DATASET_NAME="rumor/Weibo20"
CHECKPOINT="IDEA-CCNL/Erlangshen-MegatronBert-1.3B"

OPTS=""
OPTS+=" --max-length 512"
OPTS+=" --lr 5e-4"
OPTS+=" --epochs 30"
OPTS+=" --warmup-ratio 0.01"
OPTS+=" --batch-size 4"
OPTS+=" --gradient-accumulate 4"
OPTS+=" --log-iters 20"
OPTS+=" --checkpoint ${CHECKPOINT}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset-path ${DATASET_PATH}"
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --save ${SAVE_PATH}/${CONFIG}/${DATASET_NAME}/${CHECKPOINT}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/downstream/deberta_fine_tune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}/${CONFIG}/${DATASET_NAME}/${CHECKPOINT}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${CONFIG}/${DATASET_NAME}/${CHECKPOINT}/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
