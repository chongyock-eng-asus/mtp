from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model_and_tokenizer(config):
    """Setup your model with HuggingFace compatibility"""

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_basename, token=config.API_KEY)
    model = AutoModelForCausalLM.from_pretrained(config.model_basename, token=config.API_KEY)

    model.config.num_masks = config.num_masks
    model.config.lora_rank = config.lora_rank
    model.config.lora_alpha = config.lora_alpha
    model.config.API_KEY = config.API_KEY
    model.config.learning_rate = config.learning_rate
    model.config.max_length = config.max_length
    model.config.num_epochs = config.num_epochs
    model.config.model_basename = config.model_basename

    # Add special tokens if needed (adapt this to your tokenizer setup)
    if not hasattr(tokenizer, 'custom_mask_token_ids'):
        # Add your custom mask tokens
        special_tokens = [f"<mask_{i}>" for i in range(config.num_masks)]
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        tokenizer.custom_mask_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer