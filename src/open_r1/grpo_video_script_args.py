from dataclasses import dataclass, field
from trl import ScriptArguments
from typing import Optional

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/kaggle/working/train.jsonl",
        metadata={"help": "Path to the training data JSON file."},
    )
    
    eval_data_path: str = field(
        default="/kaggle/working/eval.jsonl",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="/kaggle/working/dataset",
        metadata={"help": "Path to the folder containing video files."},
    )
    
    preprocessed_data_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to the preprocessed dataset directory."},
    )
    
    # GRPO arguments
    # max_prompt_length: int = field(
    #     default=512,
    #     metadata={"help": "Maximum length of prompt"},
    # )
    
    # max_completion_length: int = field(
    #     default=128,
    #     metadata={"help": "Maximum length of completion"},
    # )
    
    # num_generations: int = field(
    #     default=2,
    #     metadata={"help": "Number of generations per prompt"},
    # )
    
    # beta: float = field(
    #     default=0.1,
    #     metadata={"help": "KL penalty coefficient"},
    # )
    
    # LoRA arguments
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for training"},
    )
    
    # lora_r: int = field(
    #     default=8,
    #     metadata={"help": "LoRA attention dimension"},
    # )
    
    # lora_alpha: int = field(
    #     default=16,
    #     metadata={"help": "LoRA alpha parameter"},
    # )
    
    # lora_dropout: float = field(
    #     default=0.05,
    #     metadata={"help": "LoRA dropout value"},
    # )
