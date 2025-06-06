export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video

# Thiết lập GPU và các tùy chọn khác
export CUDA_VISIBLE_DEVICES="0,1"
export TRITON_DISABLE_BF16="0"
export AWQ_FORCE_FP16="0"

export DEBUG_MODE="true"
export LOG_PATH="/kaggle/working/TimeZero/qwen2.5_7b_vl_tg_video.txt"

accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip "127.0.0.1" \
    --main_process_port "12361" \
    src/open_r1/grpo_video.py \
    --config_path /kaggle/working/TimeZero/configs/zero3.yaml \
    --output_dir $OUTDIR \
    --model_name_or_path unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit \
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

# torchrun --nproc_per_node="4" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12361" \
#     src/open_r1/grpo_video.py \
#     --deepspeed /kaggle/working/TimeZero/scripts/zero3_offload.json \
#     --output_dir $OUTDIR \
#     --model_name_or_path unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit \
#     --preprocessed_data_path /kaggle/working/TimeZero/dataset \
#     --train_data_path kaggle/working/train.jsonl \
#     --eval_data_path kaggle/working/eval.jsonl \
#     --dataset_name tv360_video \
#     --num-generations 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --logging_steps 1 \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --data_seed 42 \
#     --gradient_checkpointing true \
#     --attn_implementation flash_attention_2 \
#     --num_train_epochs 2 \
#     --run_name $WANDB_NAME \
#     --report_to wandb \
#     --save_steps 50 \
#     --save_only_model true
