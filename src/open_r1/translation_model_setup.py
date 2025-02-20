from unsloth import FastLanguageModel, is_bfloat16_supported


def create_model_for_translation(): 
    # Custom model creation for translation with LoRA

    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B" #"DeepSeek-R1-Distill-Llama-8B-bnb-4bit" #"unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit" #"/content/drive/MyDrive/Translation/Meta_Llama_3.1_8B_Instruct_bnb_4bit_fr_itOpusBooksSplit241024/checkpoint-1774"

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        offload_buffers=True
    )


    random_state = 3407

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = random_state,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    random_state = 3407

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = random_state,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    return model