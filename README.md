# 🎯 MVP-LAM

<div align="center">

![Concept figure](assets/concept2.png)

> #### 📄 [Paper](https://arxiv.org/abs/2602.03668) | 🌐 [Project Page](https://jmsnu.github.io/)
>
> ✒️ Jung Min Lee, Dohyeok Lee, Seokhun Ju, Taehyun Cho, Jin Woo Koo, Li Zhao, Sangwoo Hong, Jungwoo Lee   
> Seoul National University, Microsoft Research Asia, Konkuk University, HodooAI Labs

</div>

### 🔥 Highlights

- **Cross-viewpoint reconstruction** trains a latent action inferred from one view to explain the future in another view, reducing reliance on viewpoint-specific cues.
- **Action-centric latent actions** achieve higher mutual information with ground-truth actions and improved action prediction, including under out-of-distribution evaluation.
- **Improved VLA pretraining**: pretraining VLAs with MVP-LAM latent actions improves downstream manipulation performance on the SIMPLER and LIBERO-Long benchmarks.

## Table of Contents

- [🎯 MVP-LAM](#-mvp-lam)
    - [🔥 Highlights](#-highlights)
  - [Table of Contents](#table-of-contents)
  - [📢 News](#-news)
  - [🎮 Getting Started](#-getting-started)
  - [🔥 Training Recipe](#-training-recipe)
    - [0️⃣ Data Preparation](#0️⃣-data-preparation)
    - [1️⃣ Latent Action Model Training](#1️⃣-latent-action-model-training)
    - [2️⃣ VLA Pretraining](#2️⃣-vla-pretraining)
    - [3️⃣ Post-training \& Evaluation](#3️⃣-post-training--evaluation)
      - [LIBERO](#libero)
      - [SimplerEnv](#simplerenv)
  - [🚀 Performance](#-performance)
    - [SIMPLER Benchmark](#simpler-benchmark)
    - [LIBERO-Long](#libero-long)
  - [📝 Citation](#-citation)
  - [Acknowledgements](#acknowledgements)

## 📢 News

- **[2026/06]** Code release of MVP-LAM. Please check it out!

## 🎮 Getting Started

1. (Optional) Create a conda environment.

```bash
conda create -n mvplam python=3.10 -y
conda activate mvplam
```

2. Install dependencies.

```bash
# Install pytorch
# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version
# Our experiments are conducted with 'torch 2.2.0 + cuda 12.1'
pip install torch torchvision

# Clone and install
git clone https://github.com/jmsnu/mvp_lam.git
cd mvp_lam
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## 🔥 Training Recipe

### 0️⃣ Data Preparation

We train MVP-LAM on time-synchronized multi-view videos from Bridge V2.

Please refer to [this script](https://github.com/moojink/rlds_dataset_mod/blob/ad83e6c0efad5823540c0f6d3a05529596ead0b5/prepare_open_x.sh) for an example of how to download datasets from OXE.

### 1️⃣ Latent Action Model Training


MVP-LAM learns discrete latent actions with a cross-viewpoint reconstruction objective. Self-viewpoint reconstruction predicts $o_{t+1}^{v}$ from $(o_t^{v}, z_t^{v})$, while cross-viewpoint reconstruction swaps latent actions across synchronized views and predicts $o_{t+1}^{v}$ from $(o_t^{v}, z_t^{\tilde v})$ for $v \neq \tilde v$.

![Architecture figure](assets/arch.png)

To train the latent action model:

```bash
cd latent_action_model

torchrun --standalone --nnodes 1 --nproc-per-node 4 main.py fit \
    --config config/lam.yaml \
    2>&1 | tee lam.log
```

### 2️⃣ VLA Pretraining

The trained latent action model generates pseudo-labels for VLA pretraining via a next-token prediction objective. Latent action indices in the VQ-VAE codebook are mapped to dedicated tokens in the LLaMA tokenizer.

To initiate pre-training, please refer to the following script or simply run `bash ./vla-scripts/train.sh`:

```bash
GPUS_PER_NODE=4
NNODES=1
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} train.py \
    --vla.type prism-dinosiglip-224px+mx-bridge \
    --run_root_dir "vla_log"
```

### 3️⃣ Post-training & Evaluation

With the pretrained generalist policy trained to plan over an embodiment-agnostic action space, we add embodiment-specific action decoder heads for downstream deployment.

#### LIBERO

> Please first download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/main)

**Training:**

1. Set the pretrained VLA and latent action model path in `vla_path` and `lam_path` of the [training config](vla-scripts/finetune_libero.py).
2. Set your local LIBERO dataset path in `data_root_dir`.
3. Start training:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune_libero.py \
    --dataset_name "libero_10_no_noops" \
    --run_root_dir "/path/to/run_dir"
```

**Evaluation:**

```bash
pip install -r experiments/robot/libero/libero_requirements.txt

python experiments/robot/libero/run_libero_eval.py \
    --task_suite_name libero_10 \
    --action_decoder_path /path/to/your/action_decoder_path.pt \
    --pretrained_checkpoint /path/to/your/libero_finetuned_model \
    --num_trials_per_task 50 \
    --seed 7
```

#### SimplerEnv

> Our SimplerEnv evaluation is based on the [official repo (maniskill3 branch)](https://github.com/simpler-env/SimplerEnv/tree/maniskill3).

1. Clone and install SimplerEnv:

```bash
git clone -b maniskill3 https://github.com/simpler-env/SimplerEnv.git
cd SimplerEnv
pip install --upgrade git+https://github.com/haosulab/ManiSkill.git
pip install -e .
```

2. Add `experiments/robot/simpler-bridge/policies/univla` to `simpler_env/policies`, and replace `simpler_env/real2sim_eval_maniskill3.py` with `experiments/robot/simpler-bridge/real2sim_eval_maniskill3.py`.

3. Run evaluation:

```bash
# See experiments/robot/simpler-bridge/eval_simpler_bridge_4task.sh for all tasks
python real2sim_eval_maniskill3.py \
    --model="univla" \
    -e "PutSpoonOnTableClothInScene-v1" \
    -s 0 --num-episodes 24 --num-envs 1 \
    --action_decoder_path /path/to/your/action_decoder.pt \
    --ckpt_path /path/to/your/finetuned_model
```

## 🚀 Performance

### SIMPLER Benchmark

Success rate (%). Best is **bolded** and second best is <u>underlined</u>.

| Task | MVP-LAM | UniVLA | LAPA | OpenVLA | Octo-Small | Octo-Base | π₀ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StackG2Y | 33.3 | 16.7 | **54.2** | 41.6 | 8.3 | 0.0 | <u>37.5</u> |
| Carrot2Plate | **66.7** | 20.8 | <u>45.8</u> | 50.0 | 33.3 | 37.5 | 33.3 |
| Spoon2Towel | <u>66.7</u> | 54.2 | **70.8** | 37.5 | 25.0 | 12.5 | 29.2 |
| Eggplant2Bask | **75.0** | <u>66.7</u> | 58.3 | 16.7 | 12.5 | 20.8 | 45.8 |
| **AVG** | **60.4** | 39.6 | <u>57.3</u> | 36.4 | 19.8 | 17.7 | 36.5 |

### LIBERO-Long

| MVP-LAM | UniVLA (Bridge) | OpenVLA | π₀ | UniVLA (OXE) |
| --- | --- | --- | --- | --- |
| **90.8** | 79.4 | 53.7 | 85.2 | 92.0 |

## 📝 Citation

If you find our code or models useful in your work, please cite our paper:

```bibtex
@misc{lee2026mvplamlearningactioncentriclatent,
  title     = {MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction},
  author    = {Jung Min Lee and Dohyeok Lee and Seokhun Ju and Taehyun Cho and Jin Woo Koo and Li Zhao and Sangwoo Hong and Jungwoo Lee},
  year      = {2026},
  eprint    = {2602.03668},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url       = {https://arxiv.org/abs/2602.03668}
}
```

## Acknowledgements

We thank [UniVLA](https://github.com/OpenDriveLab/UniVLA) and [OpenVLA](https://github.com/openvla/openvla) for their open-sourced work!
