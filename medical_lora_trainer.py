"""
Medical LoRA PEFT Training Example

This example demonstrates how to fine-tune Gemma models on medical datasets
using LoRA for parameter-efficient training. Medical datasets require special
handling due to their domain-specific terminology and formats.
"""

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

print("Welcome to Medical LoRA Training")
print("== Training on Medical Datasets for Healthcare AI ==")

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
INTERMEDIATE_CKPT_DIR = "/home/shivajid/med_intermediate_ckpt/med/"
CKPT_DIR = "/home/shivajid/ckpts/med/"
PROFILING_DIR = "/home/shivajid/profiling/med"

# Kaggle login
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

# Sample inference before LoRA tuning
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

# Test medical questions before LoRA tuning
print("\n=== Testing Base Model (Before LoRA Tuning) ===")
medical_test_questions_before = [
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    "What is the procedure for a colonoscopy?",
    "What are the side effects of aspirin?",
    "What is the difference between systolic and diastolic blood pressure?",
]

out_data_before = sampler(
    input_strings=medical_test_questions_before,
    total_generation_steps=10,  # The number of steps performed when generating a response.
)

for input_string, out_string in zip(medical_test_questions_before, out_data_before.text):
    print(f"----------------------")
    print(f"Medical Question:\n{input_string}")
    print(f"Base Model Output:\n{out_string}")

# Apply LoRA
def get_lora_model(base_model, mesh):
    lora_provider = lora.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
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

# Example 1: Medical Instruction Dataset (MedAlpaca)
print("\n=== Training on MedAlpaca Medical Instruction Dataset ===")
train_ds_medalpaca, validation_ds_medalpaca = data_lib.create_datasets(
    dataset_name='medalpaca/medical_meadow_medqa',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,  # Use instruction format
)

# Training configuration for MedAlpaca
logging_option_medalpaca = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/peft_medalpaca", flush_every_n_steps=20
)

training_config_medalpaca = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
    metrics_logging_options=logging_option_medalpaca,
)

lora_trainer_medalpaca = peft_trainer.PeftTrainer(
    lora_gemma, optax.adamw(1e-3), training_config_medalpaca
).with_gen_model_input_fn(gen_model_input_fn)

# Train on MedAlpaca
with jax.profiler.trace(os.path.join(PROFILING_DIR, "peft_medalpaca")):
    with mesh:
        lora_trainer_medalpaca.train(train_ds_medalpaca, validation_ds_medalpaca)

# Example 2: Medical Question-Answering Dataset (PubMed QA)
print("\n=== Training on PubMed QA Dataset ===")
train_ds_pubmed, validation_ds_pubmed = data_lib.create_datasets(
    dataset_name='pubmed_qa',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)

# Train on PubMed QA
logging_option_pubmed = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/peft_pubmed", flush_every_n_steps=20
)

training_config_pubmed = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
    metrics_logging_options=logging_option_pubmed,
)

lora_trainer_pubmed = peft_trainer.PeftTrainer(
    lora_gemma, optax.adamw(1e-3), training_config_pubmed
).with_gen_model_input_fn(gen_model_input_fn)

with jax.profiler.trace(os.path.join(PROFILING_DIR, "peft_pubmed")):
    with mesh:
        lora_trainer_pubmed.train(train_ds_pubmed, validation_ds_pubmed)

# Example 3: Medical Multiple Choice QA (MedMCQA)
print("\n=== Training on MedMCQA Dataset ===")
train_ds_medmcqa, validation_ds_medmcqa = data_lib.create_datasets(
    dataset_name='medmcqa',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)

# Train on MedMCQA
logging_option_medmcqa = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/peft_medmcqa", flush_every_n_steps=20
)

training_config_medmcqa = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=CKPT_DIR,
    metrics_logging_options=logging_option_medmcqa,
)

lora_trainer_medmcqa = peft_trainer.PeftTrainer(
    lora_gemma, optax.adamw(1e-3), training_config_medmcqa
).with_gen_model_input_fn(gen_model_input_fn)

with jax.profiler.trace(os.path.join(PROFILING_DIR, "peft_medmcqa")):
    with mesh:
        lora_trainer_medmcqa.train(train_ds_medmcqa, validation_ds_medmcqa)

# Test the trained medical model
print("\n=== Testing the Trained Medical Model (After LoRA Tuning) ===")

sampler_trained = sampler_lib.Sampler(
    transformer=lora_gemma,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

# Test medical questions after training
medical_test_questions_after = [
    # Medical diagnosis
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    
    # Medical procedures
    "What is the procedure for a colonoscopy?",
    "How is a heart bypass surgery performed?",
    
    # Drug information
    "What are the side effects of aspirin?",
    "How does insulin work in the body?",
    
    # Medical terminology
    "What is the difference between systolic and diastolic blood pressure?",
    "Explain what a myocardial infarction is.",
    
    # Patient care
    "What should a patient do before a blood test?",
    "How should diabetes be managed daily?",
]

out_data_after = sampler_trained(
    input_strings=medical_test_questions_after,
    total_generation_steps=50,
)

for question, answer in zip(medical_test_questions_after, out_data_after.text):
    print(f"----------------------")
    print(f"Medical Question:\n{question}")
    print(f"Trained Model Output:\n{answer}")

print("\n=== Medical Training Complete ===")
print("The model has been fine-tuned on medical datasets and should now")
print("be better at understanding and responding to medical queries.") 
