
ckpt_path="/data3/ckpt/simpler_finetuned/ours/step-020000-epoch-11-loss=0.5437.pt+simpler+b32+lr-0.0002+lora-r32+dropout-0.0--image_aug=w-LowLevelDecoder-ws-12"
action_decoder_path="/data3/ckpt/simpler_finetuned/ours/step-020000-epoch-11-loss=0.5437.pt+simpler+b32+lr-0.0002+lora-r32+dropout-0.0--image_aug=w-LowLevelDecoder-ws-12/action_decoder-10000.pt"

export PYTHONPATH=~/vi_latent_action/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
    --model="univla" -e "PutSpoonOnTableClothInScene-v1" -s 0 --num-episodes 24 --num-envs 1 \
    --action_decoder_path ${action_decoder_path} \
    --ckpt_path ${ckpt_path} \
    
# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
#     --model="univla" -e "PutCarrotOnPlateInScene-v1" -s 0 --num-episodes 24 --num-envs 1 \
#     --action_decoder_path ${action_decoder_path} \
#     --ckpt_path ${ckpt_path} \

# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
#     --model="univla" -e "StackGreenCubeOnYellowCubeBakedTexInScene-v1" -s 0 --num-episodes 24 --num-envs 1 \
#     --action_decoder_path ${action_decoder_path} \
#     --ckpt_path ${ckpt_path} \

# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
#     --model="univla" -e "PutEggplantInBasketScene-v1" -s 0 --num-episodes 24 --num-envs 1 \
#     --action_decoder_path ${action_decoder_path} \
#     --ckpt_path ${ckpt_path} \

