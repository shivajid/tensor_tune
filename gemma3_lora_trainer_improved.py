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

print("Welcome to Improved Gemma3 4B LoRA Tuning")
print("== Optimized for Better Translation Quality ===")
print("Let's have some fun with Gemma3!")

# Data - Increased batch size for better stability
BATCH_SIZE = 32  # Increased from 16

# Model
MESH = [(2, 4), ("fsdp", "tp")]

# LoRA - Optimized for Gemma3 4B
RANK = 32  # Increased from 16 for better capacity
ALPHA = 4.0  # Increased from 2.0 for better scaling

# Train - More conservative training
MAX_STEPS = 1000  # Increased from 500
EVAL_EVERY_N_STEPS = 50  # Increased from 20
NUM_EPOCHS = 5  # Increased from 3

print(f"MESH {MESH}")
print(f"LoRA Rank: {RANK}, Alpha: {ALPHA}")
print(f"Training Steps: {MAX_STEPS}, Eval Every: {EVAL_EVERY_N_STEPS}")

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/home/shivajid/intermediate_ckpt/"
CKPT_DIR = "/home/shivajid/ckpts/"
PROFILING_DIR = "/home/shivajid/profiling/"

# Kaggle login
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()

# Download Gemma3 4B checkpoint
kaggle_ckpt_path = kagglehub.model_download("google/gemma/flax/3-4b")

# Load Gemma3 4B model
model_config = gemma3_lib.Gemma3Config.gemma3_4b()
gemma3 = gemma3_lib.Gemma3(model_config, rngs=nnx.Rngs(params=0))

checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma3)

checkpoint_path = os.path.join(INTERMEDIATE_CKPT_DIR, "state")

# If the directory exists, remove it
if os.path.exists(checkpoint_path):
    print(f"Removing existing checkpoint directory: {checkpoint_path}")
    shutil.rmtree(checkpoint_path)

checkpointer.save(os.path.join(checkpoint_path), state)
checkpointer.wait_until_finished()


def get_base_model(ckpt_path):
    """Load the base Gemma3 4B model from checkpoint."""
    model_config = gemma3_lib.Gemma3Config.gemma3_4b()
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
gemma3_tokenizer = data_lib.GemmaTokenizer(
    os.path.join(kaggle_ckpt_path, "tokenizer.model")
)

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

input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
    "Translate this into French:\nHow's the weather today?\n",
]

print("=== Base Model Performance ===")
out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=10,
)

for input_string, out_string in zip(input_batch, out_data.text):
    print(f"----------------------")
    print(f"Prompt:\n{input_string}")
    print(f"Output:\n{out_string}")

## Apply LoRA with optimized configuration

def get_lora_model(base_model, mesh):
    """Apply LoRA to the base model with optimized configuration."""
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

# Load Datasets for SFT Training with optimized parameters
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='mtnt/en-fr',
    # Alternative translation datasets:
    # dataset_name='Helsinki-NLP/opus-100',  # with data_dir="de-en" for German-English
    # dataset_name='Helsinki-NLP/opus-100',  # with data_dir="es-en" for Spanish-English
    
    global_batch_size=BATCH_SIZE,
    max_target_length=512,  # Increased from 256 for better context
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

# Training with optimized hyperparameters
logging_option = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/peft_gemma3_improved", flush_every_n_steps=50
)

# Optimized learning rate schedule for Gemma3 4B
def create_optimizer():
    """Create an optimized learning rate schedule for Gemma3 4B."""
    # Warmup for first 10% of training
    warmup_steps = MAX_STEPS // 10
    # Cosine decay for the rest
    decay_steps = MAX_STEPS - warmup_steps
    
    # Lower initial learning rate for stability
    initial_lr = 5e-4  # Reduced from 1e-3
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=initial_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=1e-5,
    )
    
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=0.01,  # Add weight decay for regularization
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    )
    
    return optimizer

# PEFT Training with optimized configuration
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
    metrics_logging_options=logging_option,
)

optimizer = create_optimizer()
lora_trainer = peft_trainer.PeftTrainer(
    lora_gemma3, optimizer, training_config
).with_gen_model_input_fn(gen_model_input_fn)

print("=== Starting Optimized LoRA Training ===")
print(f"Training for {MAX_STEPS} steps with evaluation every {EVAL_EVERY_N_STEPS} steps")
print(f"LoRA Rank: {RANK}, Alpha: {ALPHA}")
print(f"Batch Size: {BATCH_SIZE}, Max Target Length: 512")

with jax.profiler.trace(os.path.join(PROFILING_DIR, "peft_gemma3_improved")):
    with mesh:
        lora_trainer.train(train_ds, validation_ds)

# Test the trained model with improved generation parameters
print("=== Testing Trained Model ===")

sampler = sampler_lib.Sampler(
    transformer=lora_gemma3,
    tokenizer=gemma3_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=512,  # Increased cache size
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

# More diverse test cases
input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
    "Translate this into French:\nHow's the weather today?\n",
    "Translate this into French:\nThe meeting is scheduled for tomorrow.\n",
    "Translate this into French:\nI love learning new languages.\n",
]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=100,  # Increased for better completion
    temperature=0.7,  # Add temperature for better diversity
    top_p=0.9,  # Add top_p sampling
)

for input_string, out_string in zip(input_batch, out_data.text):
    print(f"----------------------")
    print(f"Prompt:\n{input_string}")
    print(f"Output:\n{out_string}")
    print(f"----------------------") 