export RUN_DATE=$(date +%Y%m%d)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 main.py fit \
    --config config/lam.yaml \
    2>&1 | tee lam.log
