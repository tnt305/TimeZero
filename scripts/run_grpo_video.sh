
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video

# Thêm các biến môi trường mới
export CUDA_VISIBLE_DEVICES="0,1"
export TRITON_DISABLE_BF16="1"
export AWQ_FORCE_FP16="1"

export DEBUG_MODE="true"
export LOG_PATH="./qwen2.5_7b_vl_tg_video.txt"
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    --gradient_accumulation_steps=2 \
    src/open_r1/grpo_video.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --preprocessed_data_path ./dataset \
    --train_data_path kaggle/working/tv360_train.jsonl \
    --eval_data_path kaggle/working/tv360_eval.jsonl \
    --dataset_name tv360_video \
    --fp16 \
    --torch_dtype float16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --num_train_epochs 2 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --save_steps 50 \
    --save_only_model true

# torchrun --nproc_per_node="2" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12361" \
#     src/open_r1/grpo_video.py \
#     --deepspeed /scripts/zero3_offload.json \
#     --output_dir $OUTDIR \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --preprocessed_data_path ./dataset \
#     --train_data_path kaggle/working/tv360_train.jsonl \
#     --eval_data_path kaggle/working/tv360_eval.jsonl \
#     --dataset_name tv360_video \
#     --num-generations 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --logging_steps 1 \
#     --fp16 \
#     --torch_dtype float16 \
#     --data_seed 42 \
#     --gradient_checkpointing true \
#     --attn_implementation eager \
#     --num_train_epochs 2 \
#     --run_name $WANDB_NAME \
#     --report_to wandb \
#     --save_steps 50 \
#     --save_only_model true
