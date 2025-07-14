# Dataset Options for LoRA PEFT Training

This guide provides comprehensive information about different datasets you can use with the Tunix LoRA PEFT trainer, including both translation and non-translation datasets.

## Current Supported Datasets

### Translation Datasets

#### 1. **MTNT (Machine Translation of Noisy Text)**
- **Dataset**: `mtnt/en-fr`
- **Description**: English to French translation dataset
- **Format**: Source (`src`) and destination (`dst`) fields
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='mtnt/en-fr',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
)
```

#### 2. **OPUS-100**
- **Dataset**: `Helsinki-NLP/opus-100`
- **Description**: Large-scale parallel corpus with 100+ language pairs
- **Available Language Pairs**:
  - `en-fr` (English-French)
  - `en-de` (English-German)
  - `en-es` (English-Spanish)
  - `en-it` (English-Italian)
  - `en-pt` (English-Portuguese)
  - And many more...
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='Helsinki-NLP/opus-100',
    # Add data_dir parameter for specific language pairs
    # data_dir="en-de" for German-English
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
)
```

### Non-Translation Datasets

#### 1. **Question-Answering Datasets**

##### SQuAD (Stanford Question Answering Dataset)
- **Dataset**: `squad`
- **Description**: Reading comprehension dataset with questions and answers
- **Format**: `question` and `answer` fields
- **Size**: ~100K examples
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='squad',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,  # Use instruction format
)
```

##### HotpotQA
- **Dataset**: `hotpot_qa`
- **Description**: Multi-hop reasoning questions requiring multiple documents
- **Format**: `question` and `answer` fields
- **Size**: ~113K examples
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='hotpot_qa',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)
```

#### 2. **Text Summarization Datasets**

##### CNN/DailyMail
- **Dataset**: `cnn_dailymail`
- **Description**: News articles with human-written summaries
- **Format**: `text` and `summary` fields
- **Size**: ~300K examples
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='cnn_dailymail',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)
```

##### XSum (Extreme Summarization)
- **Dataset**: `xsum`
- **Description**: BBC articles with single-sentence summaries
- **Format**: `text` and `summary` fields
- **Size**: ~226K examples
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='xsum',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)
```

#### 3. **Instruction Following Datasets**

##### Alpaca
- **Dataset**: `tatsu-lab/alpaca`
- **Description**: Instruction-following dataset with diverse tasks
- **Format**: `instruction` and `response` fields
- **Size**: ~52K examples
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='tatsu-lab/alpaca',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)
```

##### Dolly
- **Dataset**: `databricks/databricks-dolly-15k`
- **Description**: High-quality instruction dataset
- **Format**: `instruction` and `response` fields
- **Size**: ~15K examples
- **Usage**:
```python
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name='databricks/databricks-dolly-15k',
    global_batch_size=BATCH_SIZE,
    max_target_length=256,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)
```

## Dataset Format Requirements

### Translation Datasets
- **MTNT Format**: `{"src": "English text", "dst": "French text"}`
- **OPUS-100 Format**: `{"translation": {"en": "English text", "fr": "French text"}}`

### Non-Translation Datasets
- **QA Format**: `{"question": "What is...?", "answer": "The answer is..."}`
- **Summarization Format**: `{"text": "Long article...", "summary": "Short summary"}`
- **Instruction Format**: `{"instruction": "Write a poem...", "response": "Here's a poem..."}`

## Input Templates

The system uses different input templates based on the `instruct_tuned` parameter:

### Standard Template (instruct_tuned=False)
```python
INPUT_TEMPLATE = {
    "prefix": "Translate this into French:\n",
    "suffix": "\n",
}
```

### Instruction Template (instruct_tuned=True)
```python
INPUT_TEMPLATE_IT = {
    "prefix": "<start_of_turn>user\nTranslate this into French:\n",
    "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
}
```

## Customizing Input Templates

You can create custom input templates for different tasks:

```python
# For question-answering
qa_template = {
    "prefix": "Answer this question:\n",
    "suffix": "\n",
}

# For summarization
summary_template = {
    "prefix": "Summarize this text:\n",
    "suffix": "\n",
}

# For instruction following
instruction_template = {
    "prefix": "### Instruction:\n",
    "suffix": "\n\n### Response:\n",
}
```

## Adding New Datasets

To add support for a new dataset:

1. **Add dataset loading logic** in `create_datasets()`:
```python
elif dataset_name == "your-dataset":
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
```

2. **Add tokenization logic** in `_Tokenize.map()`:
```python
elif "your_field" in element.keys():
    src_tokens = self._tokenizer.tokenize(
        element["your_field"],
        prefix=self._input_template["prefix"],
        suffix=self._input_template["suffix"],
        add_eos=False,
    )
    dst_tokens = self._tokenizer.tokenize(
        element["target_field"], add_eos=True
    )
```

## Recommended Datasets by Task

### For Translation
1. **OPUS-100** - Large, high-quality parallel corpus
2. **MTNT** - Good for noisy text translation

### For Question-Answering
1. **SQuAD** - Standard QA benchmark
2. **HotpotQA** - Multi-hop reasoning

### For Summarization
1. **CNN/DailyMail** - News summarization
2. **XSum** - Extreme summarization

### For Instruction Following
1. **Alpaca** - Diverse instruction tasks
2. **Dolly** - High-quality instructions

## Performance Considerations

- **Dataset Size**: Larger datasets require more training time
- **Sequence Length**: Longer sequences need more memory
- **Batch Size**: Adjust based on your hardware
- **Epochs**: More epochs = better performance but longer training

## Example Usage

See `examples/non_translation_lora_trainer.py` for a complete example showing how to train on different non-translation datasets. 