import json
import gc
import os
import time
import shutil
from flax import nnx
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
from qwix import lora
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

print("Welcome to Corrected Gemma3 1B IT LoRA Tuning")
print("== Properly Handling Instruction-Tuned Model with Correct Format ===")
print("Let's have some fun with Gemma3!")

# Data - Optimized for instruction-tuned model
BATCH_SIZE = 8  # Very conservative for IT model

# Model
MESH = [(2, 4), ("fsdp", "tp")]

# LoRA - Very conservative for IT model
RANK = 8  # Very small for IT model
ALPHA = 1.0  # Very small for IT model

# Train - Very conservative training for IT model
MAX_STEPS = 100  # Very few steps for IT model
EVAL_EVERY_N_STEPS = 20  # Frequent evaluation
NUM_EPOCHS = 1  # Single epoch for IT model

print(f"MESH {MESH}")
print(f"LoRA Rank: {RANK}, Alpha: {ALPHA}")
print(f"Training Steps: {MAX_STEPS}, Eval Every: {EVAL_EVERY_N_STEPS}")

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/home/shivajid/gemma3/1b-it/intermediate_ckpt/"
CKPT_DIR = "/home/shivajid/gemma3/1b-it/ckpts_corrected/"
PROFILING_DIR = "/home/shivajid/gemma3/1b-it/profiling/"

def chk_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

chk_mkdir(INTERMEDIATE_CKPT_DIR)
chk_mkdir(CKPT_DIR)
chk_mkdir(PROFILING_DIR)

# Clean up existing checkpoints to avoid conflicts
def cleanup_checkpoints():
    """Clean up existing checkpoints to avoid parameter conflicts."""
    if os.path.exists(CKPT_DIR):
        print(f"Removing existing checkpoint directory: {CKPT_DIR}")
        shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)

#cleanup_checkpoints()

# Kaggle login
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()

# Download Gemma3 1B IT checkpoint
kaggle_ckpt_path = kagglehub.model_download("google/gemma-3/flax/gemma3-1b-it/1")

# Load Gemma3 1B IT model
model_config = gemma3_lib.Gemma3Config.gemma3_1b()

# Create the model with proper weights from the IT checkpoint
def load_it_model_with_weights():
    """Load the IT model with actual instruction-tuned weights."""
    # Load the actual weights from the IT checkpoint
    it_checkpoint_path = os.path.join(kaggle_ckpt_path, "gemma3-1b-it")
    print(f"Loading IT weights from: {it_checkpoint_path}")
    
    # Use the proper loading function from params_lib
    gemma3 = params_lib.create_model_from_checkpoint(
        checkpoint_path=it_checkpoint_path,
        model_config=model_config,
        mesh=None  # We'll shard it later
    )
    
    return gemma3

# Load the model with actual IT weights
gemma3 = load_it_model_with_weights()


checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma3)

checkpoint_path = os.path.join(INTERMEDIATE_CKPT_DIR, "state")

# Save checkpoint if it doesn't exist
#if not os.path.exists(checkpoint_path):
shutil.rmtree(checkpoint_path)
print(f"Saving checkpoint to: {checkpoint_path}")
checkpointer.save(os.path.join(checkpoint_path), state)
checkpointer.wait_until_finished()

def get_base_model(ckpt_path):
    """Load the base Gemma3 1B IT model from checkpoint."""
    model_config = gemma3_lib.Gemma3Config.gemma3_1b()
    mesh = jax.make_mesh(*MESH)
    abs_gemma3: nnx.Module = nnx.eval_shape(
        lambda: gemma3_lib.Gemma3(model_config, rngs=nnx.Rngs(params=0))
    )
    abs_state = nnx.state(abs_gemma3)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(ckpt_path, target=abs_state)

    graph_def, _ = nnx.split(abs_gemma3)
    gemma3 = nnx.merge(graph_def, restored_params)
    return gemma3, mesh, model_config

# Base model
gemma3, mesh, model_config = get_base_model(
    ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
)

# Load Gemma3 tokenizer
print(f"Gemma3 tokenizer: {params_lib.GEMMA3_TOKENIZER}")
gemma3_tokenizer = params_lib.create_tokenizer(params_lib.GEMMA3_TOKENIZER)

# Test base model performance
sampler = sampler_lib.Sampler(
    transformer=gemma3,
    tokenizer=gemma3_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

# Use the correct instruction format from params_lib
def format_instruction(prompt):
    """Format prompt using the official Gemma3 IT template."""
    return params_lib.PROMPT_TEMPLATE.format(prompt)

# Test prompts with correct instruction format
input_batch = [
    format_instruction("Tell me a story about a cow and a squirrel."),
    format_instruction("Translate this into French: Hello, my name is Morgane."),
    format_instruction("Translate this into French: This dish is delicious!"),
    format_instruction("Translate this into French: I am a student."),
    format_instruction("Translate this into French: How's the weather today?"),
]

print("=== Base Model Performance ===")
out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=50,
)

for i, (input_string, out_string) in enumerate(zip(input_batch, out_data.text)):
    print(f"----------------------")
    print(f"Test {i+1}:")
    print(f"Prompt:\n{input_string}")
    print(f"Output:\n{out_string}")


def get_lora_model(base_model, mesh):
    """Apply LoRA to the base model."""
    lora_provider = lora.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
        # comment the two args below for LoRA (w/o quantisation).
        # weight_qtype="nf4",
        # tile_size=256,
    )

    # Create dummy model input for Gemma3 (similar to test_common.py)
    dummy_model_input = {
        'last_tokens': jnp.ones((2, 1), dtype=jnp.int32),
        'positions': jnp.ones((2, 1), dtype=jnp.int32),
        'cache': None,
        'attention_mask': jnp.ones((2, 1, 1), dtype=jnp.bool_),
    }

    lora_model = lora.apply_lora_to_model(
        base_model, lora_provider, **dummy_model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model

lora_gemma3 = get_lora_model(gemma3, mesh=mesh)


# Load Datasets for SFT Training
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='mtnt/en-fr',
    # Uncomment the line below to use a Hugging Face dataset.
    # Note that this requires upgrading the 'datasets' package and restarting
    # the Colab runtime.
    # dataset_name='Helsinki-NLP/opus-100',

    # Alternative translation datasets:
    # dataset_name='Helsinki-NLP/opus-100',  # with data_dir="de-en" for German-English
    # dataset_name='Helsinki-NLP/opus-100',  # with data_dir="es-en" for Spanish-English

    # Non-translation datasets (requires code modifications):
    # dataset_name='squad',  # Question-Answering
    # dataset_name='hotpot_qa',  # Multi-hop QA
    # dataset_name='cnn_dailymail',  # Text Summarization
    # dataset_name='xsum',  # Extreme Summarization
    # dataset_name='tatsu-lab/alpaca',  # Instruction Following
    # dataset_name='databricks/databricks-dolly-15k',  # Instruction Following

    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma3_tokenizer,
)

def gen_model_input_fn(x: peft_trainer.TrainingInput):
    """Generate model inputs from training data."""
    pad_mask = x.input_tokens != gemma3_tokenizer.pad_id()
    positions = gemma_lib.build_positions_from_mask(pad_mask)
    attention_mask = gemma_lib.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

# PEFT Training
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
)

lora_trainer = peft_trainer.PeftTrainer(
    lora_gemma3, optax.adamw(1e-3), training_config
).with_gen_model_input_fn(gen_model_input_fn)

with mesh:
    lora_trainer.train(train_ds, validation_ds)

sampler = sampler_lib.Sampler(
    transformer=lora_gemma3,
    tokenizer=gemma3_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

# Use the correct instruction format from params_lib
def format_instruction(prompt):
    """Format prompt using the official Gemma3 IT template."""
    return params_lib.PROMPT_TEMPLATE.format(prompt)

# Test prompts with correct instruction format
input_batch = [
    format_instruction("Tell me a story about a cow and a squirrel."),
    format_instruction("Translate this into French: Hello, my name is Morgane."),
    format_instruction("Translate this into French: This dish is delicious!"),
    format_instruction("Translate this into French: I am a student."),
    format_instruction("Translate this into French: How's the weather today?"),
]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=50,
)

for i, (input_string, out_string) in enumerate(zip(input_batch, out_data.text)):
    print(f"----------------------")
    print(f"Test {i+1}:")
    print(f"Prompt:\n{input_string}")
    print(f"Output:\n{out_string}")




# ==============================================================================
# ADD THIS CODE TO THE END OF YOUR SCRIPT TO SAVE IN SAFETENSORS FORMAT
# ==============================================================================
print("\nConverting and saving the fine-tuned model to .safetensors format...")

# --- 1. Define the output directory ---
# This is where your final .safetensors file will be saved.
SERVABLE_CKPT_DIR = "/home/shivajid/safetensors_ckpt/gemma_1b_it_en_fr/"
if os.path.exists(SERVABLE_CKPT_DIR):
    shutil.rmtree(SERVABLE_CKPT_DIR)
os.makedirs(SERVABLE_CKPT_DIR, exist_ok=True)


# --- 2. Define the corrected conversion function with updated LoRA weight names ---
def model_state_to_hf_weights(state: nnx.State, rank: int, alpha: float) -> dict:
    """
    Converts a JAX/NNX Gemma state with LoRA adapters to a merged,
    Hugging Face compatible weight dictionary.

    This version uses the correct LoRA weight names found in your state file.
    """
    weights_dict = {}
    cpu = jax.devices("cpu")[0]
    scaling_factor = alpha / rank

    # Handle the embedding and final normalization weights
    vocab_size = 262144  # Standard Gemma vocab size
    weights_dict['model.embed_tokens.weight'] = jax.device_put(
        state.embedder.input_embedding.value, cpu
    )[:vocab_size, :]
    weights_dict['model.norm.weight'] = jax.device_put(state.final_norm.scale.value, cpu)

    # Gemma3 1B model dimensions
    embed_dim = 1152
    hidden_dim = 6 * 1152
    num_heads = 4
    num_kv_heads = 1
    head_dim = 256

    print(f"Converting model with LoRA rank={rank}, alpha={alpha}, scaling_factor={scaling_factor}")

    # Iterate through each layer to merge weights
    for idx, layer in state.layers.items():
        #print(f"layer: \n {layer} \n")
        weights_dict[f'model.layers.{idx}.input_layernorm.weight'] = jax.device_put(layer.pre_attention_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.post_attention_layernorm.weight'] = jax.device_put(layer.pre_ffw_norm.scale.value, cpu)
        # Corrected paths for q_norm and k_norm based on the provided state dump
        weights_dict[f'model.layers.{idx}.self_attn.q_norm.weight'] = jax.device_put(layer.attn._query_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.self_attn.k_norm.weight'] = jax.device_put(layer.attn._key_norm.scale.value, cpu)

        # Corrected paths for pre_feedforward_layernorm and post_feedforward_layernorm
        weights_dict[f'model.layers.{idx}.pre_feedforward_layernorm.weight'] =  jax.device_put(layer.pre_ffw_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.post_feedforward_layernorm.weight'] = jax.device_put(layer.post_ffw_norm.scale.value, cpu)




        # --- Attention Block ---
        attn = layer.attn

        # Q-Projection
        base_w_q = attn.q_einsum.w.value.transpose((0, 2, 1)).reshape((num_heads * head_dim, embed_dim))
        lora_A_q = attn.q_einsum.w_lora_a.value  # Shape: (embed_dim, rank)
        lora_B_q = attn.q_einsum.w_lora_b.value  # Shape: (rank, num_heads * head_dim)
        
        # Proper LoRA delta calculation
        delta_w_q = (lora_A_q @ lora_B_q) * scaling_factor  # Shape: (embed_dim, num_heads * head_dim)
        merged_w_q = base_w_q + delta_w_q.T  # Transpose to match base_w_q shape
        weights_dict[f'model.layers.{idx}.self_attn.q_proj.weight'] = jax.device_put(merged_w_q, cpu)

        # K-Projection: Handle KV combined tensor properly
        base_w_k = attn.kv_einsum.w.value[0].reshape(embed_dim, num_kv_heads * head_dim)
        lora_A_k = attn.kv_einsum.w_lora_a.value  # Shape: (embed_dim, rank)
        lora_B_k_raw = attn.kv_einsum.w_lora_b.value  # Shape: (rank, 2, num_kv_heads, head_dim)
        lora_B_k = lora_B_k_raw[:, 0, :, :].reshape(rank, num_kv_heads * head_dim)  # Extract K part
        
        delta_w_k = (lora_A_k @ lora_B_k) * scaling_factor
        merged_w_k = base_w_k + delta_w_k
        weights_dict[f'model.layers.{idx}.self_attn.k_proj.weight'] = jax.device_put(merged_w_k, cpu)

        # V-Projection: Handle KV combined tensor properly
        base_w_v = attn.kv_einsum.w.value[1].reshape(embed_dim, num_kv_heads * head_dim)
        lora_A_v = attn.kv_einsum.w_lora_a.value  # Shape: (embed_dim, rank)
        lora_B_v_raw = attn.kv_einsum.w_lora_b.value  # Shape: (rank, 2, num_kv_heads, head_dim)
        lora_B_v = lora_B_v_raw[:, 1, :, :].reshape(rank, num_kv_heads * head_dim)  # Extract V part
        
        delta_w_v = (lora_A_v @ lora_B_v) * scaling_factor
        merged_w_v = base_w_v + delta_w_v
        weights_dict[f'model.layers.{idx}.self_attn.v_proj.weight'] = jax.device_put(merged_w_v, cpu)

        # O-Projection: No LoRA applied, just reshape properly
        base_w_o = attn.attn_vec_einsum.w.value.reshape(embed_dim, num_heads * head_dim)
        weights_dict[f'model.layers.{idx}.self_attn.o_proj.weight'] = jax.device_put(base_w_o, cpu)

        # --- MLP Block ---
        mlp = layer.mlp

        # Gate Projection: Proper LoRA merging
        base_w_gate = mlp.gate_proj.kernel.value.T  # Shape: (hidden_dim, embed_dim)
        lora_A_gate = mlp.gate_proj.kernel_lora_a.value  # Shape: (embed_dim, rank)
        lora_B_gate = mlp.gate_proj.kernel_lora_b.value  # Shape: (rank, hidden_dim)
        
        delta_w_gate = (lora_A_gate @ lora_B_gate) * scaling_factor
        merged_w_gate = base_w_gate + delta_w_gate.T
        weights_dict[f'model.layers.{idx}.mlp.gate_proj.weight'] = jax.device_put(merged_w_gate, cpu)

        # Up Projection: Proper LoRA merging
        base_w_up = mlp.up_proj.kernel.value.T  # Shape: (hidden_dim, embed_dim)
        lora_A_up = mlp.up_proj.kernel_lora_a.value  # Shape: (embed_dim, rank)
        lora_B_up = mlp.up_proj.kernel_lora_b.value  # Shape: (rank, hidden_dim)
        
        delta_w_up = (lora_A_up @ lora_B_up) * scaling_factor
        merged_w_up = base_w_up + delta_w_up.T
        weights_dict[f'model.layers.{idx}.mlp.up_proj.weight'] = jax.device_put(merged_w_up, cpu)

        # Down Projection: Proper LoRA merging
        base_w_down = mlp.down_proj.kernel.value.T  # Shape: (embed_dim, hidden_dim)
        lora_A_down = mlp.down_proj.kernel_lora_a.value  # Shape: (hidden_dim, rank)
        lora_B_down = mlp.down_proj.kernel_lora_b.value  # Shape: (rank, embed_dim)
        
        delta_w_down = (lora_A_down @ lora_B_down) * scaling_factor
        merged_w_down = base_w_down + delta_w_down.T
        weights_dict[f'model.layers.{idx}.mlp.down_proj.weight'] = jax.device_put(merged_w_down, cpu)

    return weights_dict


# --- 3. Execute the conversion and save the file ---

# Extract the state (the weights) from the trained LoRA model object
_, lora_state = nnx.split(lora_gemma3)


# Perform the conversion on the CPU to be memory-safe
with jax.default_device(jax.devices("cpu")[0]):
    hf_weights = model_state_to_hf_weights(lora_state,rank=RANK, alpha=ALPHA)

print('Finished converting model weights to safetensors format')


# vLLM wants the weight dictionary flattened
def flatten_weight_dict(torch_params, prefix=""):
    flat_params = {}
    for key, value in torch_params.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat_params.update(flatten_weight_dict(value, new_key + "."))
        else:
            flat_params[new_key] = value
    return flat_params

servable_weights = flatten_weight_dict(hf_weights)

from safetensors.flax import save_file

# Save the converted weights dictionary to a .safetensors file
save_file(servable_weights, os.path.join(SERVABLE_CKPT_DIR, 'model.safetensors'))

print(f" Model successfully saved to {os.path.join(SERVABLE_CKPT_DIR, 'model.safetensors')}")

from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/gemma-3-1b-it", allow_patterns="*.json", local_dir=SERVABLE_CKPT_DIR)
