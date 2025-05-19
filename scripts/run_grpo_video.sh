
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video

# Thêm các biến môi trường mới
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Sử dụng cả 4 GPU L4
export TRITON_DISABLE_BF16="0"  # Cho phép sử dụng bfloat16
export AWQ_FORCE_FP16="0"  # Tắt force fp16

export DEBUG_MODE="true"
export LOG_PATH="/kaggle/working/TimeZero/qwen2.5_7b_vl_tg_video.txt"

# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --mixed_precision=bf16 \
#     --gradient_accumulation_steps=1 \
#     src/open_r1/grpo_video.py \
#     --config_file scripts/zero3_offload.json \
#     --output_dir $OUTDIR \
#     --model_name_or_path /kaggle/working/Qwen2.5-VL-3B-Instruct \
#     --preprocessed_data_path ./dataset \
#     --train_data_path kaggle/working/tv360_train.jsonl \
#     --eval_data_path kaggle/working/tv360_eval.jsonl \
#     --dataset_name tv360_video \
#     --fp16 \
#     --torch_dtype bfloat16 \
#     --data_seed 42 \
#     --gradient_checkpointing true \
#     --attn_implementation eager \
#     --run_name $WANDB_NAME \
#     --report_to wandb \
#     --save_only_model true

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12361" \
    src/open_r1/grpo_video.py \
    # --deepspeed /kaggle/working/TimeZero/scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path /kaggle/working/Qwen2-VL-2B-Instruct \
    --preprocessed_data_path /kaggle/working/TimeZero/dataset \
    --train_data_path kaggle/working/train.jsonl \
    --eval_data_path kaggle/working/eval.jsonl \
    --dataset_name tv360_video \
    --num-generations 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --save_steps 50 \
    --save_only_model true
