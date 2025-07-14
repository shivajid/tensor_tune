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
#PEFT
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
    metrics_logging_options=logging_option,
)
lora_trainer = peft_trainer.PeftTrainer(
    lora_gemma, optax.adamw(1e-3), training_config
).with_gen_model_input_fn(gen_model_input_fn)

with jax.profiler.trace(os.path.join(PROFILING_DIR, "peft")):
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
     
