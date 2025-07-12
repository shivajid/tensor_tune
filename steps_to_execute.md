# Steps to Execute PEFT Tuning with JAX

This document provides the commands to execute the steps outlined in the `README.md` for PEFT tuning with JAX.

## 1. Create a TPUVM

This step is specific to your cloud provider. For Google Cloud, you would use a command similar to this:

```bash
# This is an example command and may need to be adjusted for your project and zone.
gcloud compute tpus tpu-vm create your-tpu-name \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-vm-base
```

## 2. Create a Python Virtual Environment

After connecting to your TPU VM, you may need to install Python 3.11 if it is not available.

*Note: The following commands are for Debian/Ubuntu-based systems. You may need to adapt them for your specific OS.*
```bash
# Update package list and install python 3.11
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv
```

Once Python 3.11 is installed, create a Python 3.11 virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 3. Install Pip Dependencies

Install the necessary Python libraries. Based on `gemma_lora_trainer.py`, the following libraries are required.

```bash
pip install flax jax jax-numpy kagglehub optax orbax-checkpoint qwix tunix datasets sentencepiece
```
*Note: Additional dependencies might be required depending on the execution environment.*

## 4. Create Directories

Create the directories for checkpoints, profiling data, and metrics.

```bash
mkdir -p /home/shivajid/intermediate_ckpt/
mkdir -p /home/shivajid/ckpts/
mkdir -p /home/shivajid/profiling/
mkdir -p /tmp/tensorboard/peft
```

## 5. Run the Python Code

Execute the main training script.

```bash
python gemma_lora_trainer.py
```

## 6. Evaluate the Outputs

The script will print the output of the model before and after training. The evaluation is printed to the console. You can redirect the output to a file for further analysis.

```bash
python gemma_lora_trainer.py > training_output.log
```

## 7. Load the Metrics using XProf

If you are running this in a Google Cloud environment, you can use the XProf tool to visualize the profiling data.

First, you need to copy the profiling data to a GCS bucket:
```bash
# Make sure you have gsutil installed and configured
gsutil -m cp -r /home/shivajid/profiling/ gs://your-bucket/profiling/
```

Then you can launch a TensorBoard instance pointing to that bucket.

```bash
# Make sure you have tensorboard installed
tensorboard --logdir gs://your-bucket/profiling/
```

## 8. Repeat with a new dataset

To use a new dataset, you need to modify the `dataset_name` parameter in the `create_datasets` function call within `gemma_lora_trainer.py`.

For example, to use the `Helsinki-NLP/opus-100` dataset, you would change this line:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='mtnt/en-fr',
    ...
)
```
to:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='Helsinki-NLP/opus-100',
    # You might need to adjust other parameters like language pairs
    ...
)
```
After modifying the script, you can re-run it.
