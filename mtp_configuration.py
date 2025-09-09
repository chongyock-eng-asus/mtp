from transformers import PreTrainedModel, PretrainedConfig
import os

class MultiTokenPredictionConfig(PretrainedConfig):
    model_type = "multi_token_prediction"

    def __init__(
        self,
        model_basename: str = "allenai/Llama-3.1-Tulu-3-8B-SFT",
        datasource: str = "allenai/tulu-3-sft-mixture",
        num_masks: int = 8,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        API_KEY: str = None,
        WANDB_API_KEY: str = None,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        num_epochs: int = 1,
        output_dir:str = "mtp_model_output",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_basename = model_basename
        self.datasource = datasource
        self.num_masks = num_masks
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.API_KEY = os.getenv('API_KEY')
        self.WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.output_dir = output_dir
