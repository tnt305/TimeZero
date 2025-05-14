
export WANDB_PROJECT=Video-GRPO
export WANDB_NAME=Qwen2.5_7b_TG

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=outputs_video

export DEBUG_MODE="true"
export LOG_PATH="./qwen2.5_7b_vl_tg_video.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12361" \
    TimeZero/src/open_r1/grpo_video.py \
    --deepspeed TimeZero/scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --preprocessed_data_path ./dataset \
    --train_data_path kaggle/working/tv360_train.jsonl \
    --eval_data_path kaggle/working/tv360_eval.jsonl \
    --dataset_name tv360_video \
    --max_prompt_length 512 \
    --max_completion_length 256 \
    --num_generations 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
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
