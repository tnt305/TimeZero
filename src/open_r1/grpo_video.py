# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import re
import torch
import cv2
import json
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import (
    load_dataset, 
    load_from_disk, 
    Dataset, 
    DatasetDict
)
from transformers import (
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training,
    get_peft_model, PeftModel
)
from math_verify import parse, verify
from trl import (
    GRPOConfig, 
    GRPOTrainer, 
    ModelConfig, 
    ScriptArguments, 
    TrlParser, 
    get_peft_config
)
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from src.open_r1.grpo_video_script_args import GRPOScriptArguments
from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video as Qwen2VLGRPOTrainer
from src.open_r1.trainer import Qwen2VLGRPOVLLMTrainer_Video as Qwen2VLGRPOVLLMTrainer

def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    print('last_answer_content:', last_answer_content)

    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE)
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time

def iou_timestamp_reward(completions, solution, durations, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    # print(completions, solution, durations)
    # contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # print(completions, solution, durations, **kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, duration in zip(completions, solution, durations): # Added video_durations
        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        # s, e = gt_start / duration, gt_end / duration
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union   # 0.1 0.3

            reward = iou

        print('gt second:', gt_start, gt_end)
        print('pred second:', start_time, end_time)
        print(f"------------- {current_time} IoU reward: {reward} -------------\n")

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"Content: {content}\n")
                f.write(f"pred second: {str(start_time)}, {str(end_time)}\n")
                f.write(f"gt second: {str(gt_start)}, {str(gt_end)}\n")
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n") # Modified log message

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print('matches:', matches)
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "iou": iou_timestamp_reward, # Modified registry to use iou_timestamp_reward
    "format": format_reward,
}

QUESTION_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

def preprocess_video_inner(video_path, processor, max_pixels, min_pixels):
    messages = [
        {"role": "user", "content": [
                {"type": "video",
                "video": video_path,
                "total_pixels": max_pixels,
                "min_pixels": min_pixels,
                },
            ]
        },
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    return image_inputs, video_inputs, video_kwargs, fps_inputs

def get_duration(video_path):
    if not os.path.exists(video_path):
        # raise FileNotFoundError(f"Video file not found: {video_path}")
        print(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0

    return duration

def load_json_dataset(train_data_path, eval_data_path, preprocessed_data_path= "./dataset"): # Modified to accept preprocessed_data_path
    max_pixels = 3584 * 28 * 28
    min_pixels = 16 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        )
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        examples = []
        for video in tqdm(data):
            timestamps = video["timestamp"]
            sentence = video["caption"].strip().lower()
            if sentence.endswith("."):
                sentence = sentence[:-1]
    
            video_path = video["video"]
            video_id = video_path.split("/")[-1].split(".")[0]
            
            example_output_dir = video_path.split("/")[-1].split(".")[0]
            example_output_dir = f"./dataset/{split_name}/{example_output_dir}"
            os.makedirs(example_output_dir, exist_ok=True)

            _, video_inputs, video_kwargs, fps_inputs = preprocess_video_inner(video_path, processor, max_pixels, min_pixels)
            # Validate video inputs
            if video_inputs is None or len(video_inputs) == 0:
                print(f"Warning: No valid frames extracted from {video_path}")
                continue
            
            torch.save(video_inputs, os.path.join(example_output_dir, "video_inputs.pt"))
            with open(os.path.join(example_output_dir, "video_kwargs.json"), 'w') as f:
                json.dump(video_kwargs, f)
            
            example_output_dir = os.path.join(preprocessed_data_path, split_name, f"{video_id}")
            duration = get_duration(video["video"])
            solution = (float(timestamps[0]) / duration, float(timestamps[1]) / duration)
            example = {
                "problem": sentence,
                "solution": (timestamps[0], timestamps[1]),
                "video_path": video_path,
                "durations": duration,
                "preprocessed_path": example_output_dir # Initialize preprocessed_path as None
            }
            if preprocessed_data_path != "": # If preprocessed data path is provided, construct the path
                example["preprocessed_path"] = os.path.join(preprocessed_data_path, split_name, f"{video_id}")
            examples.append(example)
        return examples

    train_dataset = create_dataset_from_json(train_data_path, "train")
    eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})

def main(script_args, training_args, model_args):

    # # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset, now handles both raw and preprocessed data
    dataset = load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.preprocessed_data_path # Pass preprocessed_data_path
    )

    if script_args.use_lora:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj"], #"gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"
            inference_mode=False,
            r= 4,
            lora_alpha=8,
            lora_dropout=0.025,
            bias="none",
        )
    else:
        lora_config = None
    
    training_args = GRPOConfig(
        deepspeed = "./scripts/zero3_offload.json",
        fp16 = True,
        num_generations = 1,
        optim = "adamw_8bit",
        lr_scheduler_type = "cosine",
        gradient_accumulation_steps = 1,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        num_train_epochs = 1,
        
    )
    
    
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    # print("using: ", trainer_cls)
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    trainer_cls = Qwen2VLGRPOTrainer #if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("Using trainer class:", trainer_cls)

    # Prepare model for k-bit training
    # model = prepare_model_for_kbit_training(model)
    # # Apply LoRA
    # model = get_peft_model(model, lora_config)
    
    # Initialize trainer với cả LoRA và GRPO config
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=lora_config,  # Sử dụng LoRA config
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train và push model
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))

    default_args = [
            "--dataset_name", "tv360_video",
            "--deepspeed" ,"./scripts/zero3_offload.json",
            "--model_name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct",
            "--trust_remote_code", "True",
            "--fp16", "True",
            "--num_generations", "1",
            "--torch_dtype", "bfloat16",
            "--attn_implementation", "eager",
            "--optim", "adamw_8bit",
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--lr_scheduler_type", "cosine",
            "--gradient_accumulation_steps", "1",
    ]
    script_args, training_args, model_args = parser.parse_args_and_config(default_args)
    main(script_args, training_args, model_args)

