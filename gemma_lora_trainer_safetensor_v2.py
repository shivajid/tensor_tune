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
from tunix.models.gemma import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import profiler
import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

print("Welcome to qLora Tuning joy ride")
print("== Ready for the Fun Manga== joy ride yippe creatures ==")
print ("lets have some fun")
#total_devices = jax.device_count()
#print(f"Total JAX devices available: {total_devices}")

# Data
BATCH_SIZE = 16

# Model
MESH = [(1, 8), ("fsdp", "tp")]
# LoRA
RANK = 16
ALPHA = 2.0

# Train
MAX_STEPS = 500
EVAL_EVERY_N_STEPS = 20
NUM_EPOCHS = 3

print(f"MESH {MESH}")

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/home/shivajid/intermediate_ckpt/"
CKPT_DIR = "/home/shivajid/ckpts/"
PROFILING_DIR = "/home/shivajid/profiling/"


#Kaggle login
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
  kagglehub.login()


kaggle_ckpt_path = kagglehub.model_download("google/gemma/flax/2b")


params = params_lib.load_and_format_params(os.path.join(kaggle_ckpt_path, "2b"))

gemma = gemma_lib.Transformer.from_params(params, version="2b")

checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma)

checkpoint_path = os.path.join(INTERMEDIATE_CKPT_DIR, "state")

# If the directory exists, remove it
if os.path.exists(checkpoint_path):
    print(f"Removing existing checkpoint directory: {checkpoint_path}")
    shutil.rmtree(checkpoint_path)

checkpointer.save(os.path.join(checkpoint_path), state)
checkpointer.wait_until_finished()


def get_base_model(ckpt_path):

  model_config = gemma_lib.TransformerConfig.gemma_2b()
  mesh = jax.make_mesh(*MESH)
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma, mesh, model_config


# Base model
gemma, mesh, model_config = get_base_model(
    ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
)


gemma_tokenizer = data_lib.GemmaTokenizer(
    os.path.join(kaggle_ckpt_path, "tokenizer.model")
)

sampler = sampler_lib.Sampler(
    transformer=gemma,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
    "Translate this into French:\nHow's the weather today?\n",
]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=10,  # The number of steps performed when generating a response.
)

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"----------------------")
  print(f"Prompt:\n{input_string}")
  print(f"Output:\n{out_string}")



## Apply Lora

def get_lora_model(base_model, mesh):
  lora_provider = lora.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=RANK,
      alpha=ALPHA,
      # comment the two args below for LoRA (w/o quantisation).
      #weight_qtype="nf4",
      #tile_size=256,
  )

  model_input = base_model.get_model_input()
  lora_model = lora.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model

lora_gemma = get_lora_model(gemma, mesh=mesh)

#Load Datasets for SFT Training
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
    tokenizer=gemma_tokenizer,
)

def gen_model_input_fn(x: peft_trainer.TrainingInput):
  pad_mask = x.input_tokens != gemma_tokenizer.pad_id()
  positions = gemma_lib.build_positions_from_mask(pad_mask)
  attention_mask = gemma_lib.make_causal_attn_mask(pad_mask)
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': positions,
      'attention_mask': attention_mask,
  }

#Training with full weights
logging_option = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/peft", flush_every_n_steps=20
)

'''
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    metrics_logging_options=logging_option,
)
trainer = peft_trainer.PeftTrainer(gemma, optax.adamw(1e-5), training_config)
trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

with jax.profiler.trace(os.path.join(PROFILING_DIR, "full_trainingi_2")):
  with mesh:
    trainer.train(train_ds, validation_ds)
'''

profiler_option = profiler.ProfilerOptions(log_dir=PROFILING_DIR, skip_first_n_steps=2, profiler_steps=25)

#PEFT
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
    metrics_logging_options=logging_option,
    profiler_options=profiler_option,
)
lora_trainer = peft_trainer.PeftTrainer(
    lora_gemma, optax.adamw(1e-3), training_config
).with_gen_model_input_fn(gen_model_input_fn)

with mesh:
    lora_trainer.train(train_ds, validation_ds)

gemma_tokenizer = data_lib.GemmaTokenizer(
    os.path.join(kaggle_ckpt_path, "tokenizer.model")
)

sampler = sampler_lib.Sampler(
    transformer=lora_gemma,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)


input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
    "Translate this into French:\nHow's the weather today?\n",
]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=50,  # The number of steps performed when generating a response.
)

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"----------------------")
  print(f"Prompt:\n{input_string}")
  print(f"Output:\n{out_string}")



## generate
print ("Merging checkpoint")
print ("Loading checkpoint")

trained_ckpt_path = os.path.join(CKPT_DIR, "500", "model_params")

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_gemma, nnx.LoRAParam),
)

print("restoring checkpoint")

checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)



nnx.update(
    lora_gemma,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_gemma, nnx.LoRAParam),
        trained_lora_params,
    ),
)

gemma_tokenizer = data_lib.GemmaTokenizer(
    os.path.join(kaggle_ckpt_path, "tokenizer.model")
)
#sampler = sampler_lib.Sampler(
#    transformer=lora_gemma, tokenizer=gemma_tokenizer
#)

input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
    "Translate this into French:\nHow's the weather today?\n",
]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=10,  # The number of steps performed when generating a response.
)

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"----------------------")
  print(f"Prompt:\n{input_string}")
  print(f"Output:\n{out_string}")

# ==============================================================================
# ADD THIS CODE TO THE END OF YOUR SCRIPT TO SAVE IN SAFETENSORS FORMAT
# ==============================================================================
print("\nConverting and saving the fine-tuned model to .safetensors format...")

# --- 1. Define the output directory ---
# This is where your final .safetensors file will be saved.
SERVABLE_CKPT_DIR = "/home/shivajid/safetensors_ckpt/v3/"
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
    vocab_size = 256000 # Standard Gemma vocab size
    weights_dict['model.embed_tokens.weight'] = jax.device_put(
        state.embedder.input_embedding.value, cpu
    )[:vocab_size, :]
    weights_dict['model.norm.weight'] = jax.device_put(state.final_norm.scale.value, cpu)

    # Gemma 2B model dimensions
    embed_dim = 2048
    hidden_dim = 16384
    num_heads = 8
    num_kv_heads = 1
    head_dim = 256

    # Iterate through each layer to merge weights
    for idx, layer in state.layers.items():
        weights_dict[f'model.layers.{idx}.input_layernorm.weight'] = jax.device_put(layer.pre_attention_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.post_attention_layernorm.weight'] = jax.device_put(layer.pre_ffw_norm.scale.value, cpu)

        # --- Attention Block ---
        attn = layer.attn

        # Q-Projection
        base_w_q = attn.q_einsum.w.value.transpose((0, 2, 1)).reshape((num_heads * head_dim, embed_dim))
        lora_A_q = attn.q_einsum.w_lora_a.value
        lora_B_q_raw = attn.q_einsum.w_lora_b.value
        print(f"Lora shape lora_B_q_raw: {lora_B_q_raw.shape}")
        lora_B_q = lora_B_q_raw.reshape(rank, -1)

        print(f"Lora shape Lora_B_q: {lora_B_q.shape}")
        print(f"Lora shape Lora_A_q: {lora_A_q.shape}")
        print(f"Base shape base_w_q: {base_w_q.shape}")
        delta_w_q = (lora_A_q @ lora_B_q) * scaling_factor
        weights_dict[f'model.layers.{idx}.self_attn.q_proj.weight'] = jax.device_put(base_w_q + delta_w_q, cpu)

        # K-Projection
        base_w_k = attn.kv_einsum.w.value[0].reshape(embed_dim, -1).T
        lora_A_k = attn.kv_einsum.w_lora_a.value
        lora_B_k_raw = attn.kv_einsum.w_lora_b.value
        print(f"Lora shape base_w_k: {base_w_k.shape}")
        print(f"Lora shape lora_A_k: {lora_A_k.shape}")
        print(f"Lora shape lora_B_k_raw: {lora_B_k_raw.shape}")

        #lora_B_k = lora_B_k_raw.squeeze(0).T
        lora_B_k = lora_B_k_raw[:, 0, :, :].squeeze()
        print(f"Corrected Lora shape lora_B_k: {lora_B_k.shape}")
        #print(f"Lora shape lora_B_k: {lora_B_k}")
        delta_w_k = (lora_A_k @ lora_B_k) * scaling_factor
        delta_w_k_T = delta_w_k.T
        weights_dict[f'model.layers.{idx}.self_attn.k_proj.weight'] = jax.device_put(base_w_k + delta_w_k_T, cpu)

        # V-Projection
        base_w_v = attn.kv_einsum.w.value[1].reshape(embed_dim, -1).T
        lora_A_v = attn.kv_einsum.w_lora_a.value
        lora_B_v_raw = attn.kv_einsum.w_lora_b.value
        print(f"Lora shape base_w_v: {base_w_v.shape}")
        print(f"Lora shape lora_A_v: {lora_A_v.shape}")
        print(f"Lora shape lora_B_v_raw: {lora_B_v_raw.shape}")

        #lora_B_v = lora_B_v_raw.squeeze(0).T
        lora_B_v = lora_B_v_raw[:, 0, :, :].squeeze()
        print(f"Corrected Lora shape lora_B_k: {lora_B_k.shape}")
        print(f"Lora shape lora_B_v: {lora_B_v.shape}")
        delta_w_v = (lora_A_v @ lora_B_v) * scaling_factor
        delta_w_v_T = delta_w_v.T
        weights_dict[f'model.layers.{idx}.self_attn.v_proj.weight'] = jax.device_put(base_w_v + delta_w_v.T, cpu)

        # O-Projection (No LoRA applied in your config)
        base_w_o = attn.attn_vec_einsum.w.value.reshape(embed_dim, embed_dim).T
        weights_dict[f'model.layers.{idx}.self_attn.o_proj.weight'] = jax.device_put(base_w_o, cpu)

        # --- MLP Block ---
        mlp = layer.mlp

        # Gate Projection
        base_w_gate = mlp.gate_proj.kernel.value.T
        lora_A_gate = mlp.gate_proj.kernel_lora_a.value
        lora_B_gate_raw = mlp.gate_proj.kernel_lora_b.value
        print(f"Lora shape base_w_gate: {base_w_gate.shape}")
        print(f"Lora shape lora_A_gate: {lora_A_gate.shape}")
        print(f"Lora shape lora_B_gate_raw: {lora_B_gate_raw.shape}")


        lora_B_gate = lora_B_gate_raw
        print(f"Lora shape lora_B_gate: {lora_B_gate.shape}")
        delta_w_gate = (lora_A_gate @ lora_B_gate) * scaling_factor
        delta_w_gate_T = delta_w_gate.T
        weights_dict[f'model.layers.{idx}.mlp.gate_proj.weight'] = jax.device_put(base_w_gate + delta_w_gate.T, cpu)

        # Up Projection
        base_w_up = mlp.up_proj.kernel.value.T
        lora_A_up = mlp.up_proj.kernel_lora_a.value
        lora_B_up_raw = mlp.up_proj.kernel_lora_b.value
        print(f"Lora shape base_w_up: {base_w_up.shape}")
        print(f"Lora shape lora_A_up: {lora_A_up.shape}")
        print(f"Lora shape lora_B_up_raw: {lora_B_up_raw.shape}")


        lora_B_up = lora_B_up_raw
        print(f"Lora shape lora_B_up.T: {lora_B_up.shape}")
        delta_w_up = (lora_A_up @ lora_B_up) * scaling_factor
        delta_w_up_T = delta_w_up.T
        weights_dict[f'model.layers.{idx}.mlp.up_proj.weight'] = jax.device_put(base_w_up + delta_w_up_T, cpu)

        # Down Projection
        base_w_down = mlp.down_proj.kernel.value.T
        lora_A_down = mlp.down_proj.kernel_lora_a.value
        lora_B_down_raw = mlp.down_proj.kernel_lora_b.value
        print(f"Lora shape base_w_down: {base_w_down.shape}")
        print(f"Lora shape lora_A_down: {lora_A_down.shape}")
        print(f"Lora shape lora_B_down_raw: {lora_B_down_raw.shape}")


        lora_B_down = lora_B_down_raw
        print(f"Lora shape lora_B_down.T: {lora_B_down.shape}")
        delta_w_down = (lora_A_down @ lora_B_down) * scaling_factor
        delta_w_down_T = delta_w_down.T
        weights_dict[f'model.layers.{idx}.mlp.down_proj.weight'] = jax.device_put(base_w_down + delta_w_down_T, cpu)

    return weights_dict


# --- 3. Execute the conversion and save the file ---

# Extract the state (the weights) from the trained LoRA model object
_, lora_state = nnx.split(lora_gemma)


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

#from huggingface_hub import snapshot_download
#snapshot_download(repo_id="google/gemma-2b", allow_patterns="*.json", local_dir=SERVABLE_CKPT_DIR)


