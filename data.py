# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loading and preprocessing."""

from collections.abc import Iterable
from typing import Any

import datasets
from etils import epath
from grain import python as grain
import numpy as np
import tensorflow_datasets as tfds
from tunix.sft.peft_trainer import TrainingInput  # pylint: disable=g-importing-member

import sentencepiece as spm

INPUT_TEMPLATE = {
    "prefix": "Translate this into French:\n",
    "suffix": "\n",
}

INPUT_TEMPLATE_IT = {
    "prefix": "<start_of_turn>user\nTranslate this into French:\n",
    "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
}


class GemmaTokenizer(spm.SentencePieceProcessor):
  """Tokenizing and encoding/decoding text using the Sentencepiece tokenizer."""

  _GEMMA2_TOKENIZER_PATH: epath.PathLike = (
      'gs://gemma-data/tokenizers/tokenizer_gemma2.model'
  )

  def __init__(self, model_path: str = _GEMMA2_TOKENIZER_PATH):
    model_proto = epath.Path(model_path).read_bytes()
    super().__init__()
    self.LoadFromSerializedProto(model_proto)

  def tokenize(
      self,
      example: str,
      prefix: str = "",
      suffix: str = "",
      add_eos: bool = True,
  ) -> np.ndarray:
    """The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
        sequence.

    Returns:
      Tokens corresponding to the input string.
    """
    int_list = [self.bos_id()]
    int_list.extend(self.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self.eos_id())

    return np.array(int_list, dtype=np.int32)


def create_datasets(
    dataset_name: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: GemmaTokenizer,
    instruct_tuned: bool = False,
    input_template: dict[str, str] | None = None,
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:
  """Creates train and eval data iterator.

  Args:
    dataset_name: The name of the dataset to use.
    global_batch_size: The global batch size to use for both train and eval.
    max_target_length: The maximum length of the target sequence.
    num_train_epochs: The number of epochs to use for training. If None, the
      dataset will be repeated indefinitely.
    tokenizer: The tokenizer to use for tokenizing the dataset.
    instruct_tuned: Whether the dataset should be instruct tuned.
    input_template: The input template to use for the dataset.

  Returns:
    A tuple of train and eval data iterators.
  """
  if dataset_name == "mtnt/en-fr":
    train_ds, eval_ds = tfds.data_source(dataset_name, split=("train", "valid"))
  elif dataset_name == "Helsinki-NLP/opus-100":  # Hugging Face dataloader
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, data_dir="en-fr", split=("train", "validation")
    )
  elif dataset_name == "squad":  # Question-Answering
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
  elif dataset_name == "hotpot_qa":  # Multi-hop QA
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
  elif dataset_name == "cnn_dailymail":  # Summarization
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, "3.0.0", split=("train", "validation")
    )
  elif dataset_name == "xsum":  # Extreme Summarization
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
  elif dataset_name == "tatsu-lab/alpaca":  # Instruction following
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
  elif dataset_name == "databricks/databricks-dolly-15k":  # Instruction following
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
  # Medical Datasets
  elif dataset_name == "medalpaca/medical_meadow_medqa":  # Medical instruction dataset
    train_ds = datasets.load_dataset(
        dataset_name, split=("train")
    )
    split_dataset = train_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_dataset['train']
    eval_ds = split_dataset['test']

    #eval_ds = None
  elif dataset_name == "microsoft/DialoGPT-medium":  # Medical dialogue (can be adapted)
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
  elif dataset_name == "pubmed_qa":  # Medical QA dataset
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, split=("train", "validation")
    )
  elif dataset_name == "medical_dialog":  # Medical dialogue dataset
    train_ds, eval_ds = datasets.load_dataset(
        "medical_dialog", split=("train", "validation")
    )
  elif dataset_name == "medqa":  # Medical question answering
    train_ds, eval_ds = datasets.load_dataset(
        "medqa", split=("train", "validation")
    )
  elif dataset_name == "medmcqa":  # Medical multiple choice QA
    train_ds, eval_ds = datasets.load_dataset(
        "medmcqa", split=("train", "validation")
    )
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

  input_template = INPUT_TEMPLATE_IT if instruct_tuned else INPUT_TEMPLATE
  eval_loader = None
  
  train_loader = _build_data_loader(
      data_source=train_ds,
      batch_size=global_batch_size,
      num_epochs=num_train_epochs,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  if eval_ds is not None:
   eval_loader = _build_data_loader(
      data_source=eval_ds,
      batch_size=global_batch_size,
      num_epochs=1,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  return train_loader, eval_loader


def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: GemmaTokenizer,
    input_template: dict[str, str],
) -> grain.DataLoader:
  """Builds a data loader for the given data source."""
  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=[
          _Tokenize(tokenizer, input_template),
          _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
          _FilterOverlength(max_seq_len),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
  )


class _Tokenize(grain.MapTransform):
  """Tokenize the input."""

  def __init__(self, tokenizer: GemmaTokenizer, input_template: dict[str, str]):
    self._tokenizer = tokenizer
    self._input_template = input_template

  def map(self, element: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize the input."""
    if "src" in element.keys():  ## MTNT dataset
      src_tokens = self._tokenizer.tokenize(
          element["src"].decode(),
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["dst"].decode(), add_eos=True
      )
    elif "translation" in element.keys():  ## OPUS-100 dataset
      src_tokens = self._tokenizer.tokenize(
          element["translation"]["en"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["translation"]["fr"], add_eos=True
      )
    elif "question" in element.keys() and "answer" in element.keys():  ## QA datasets
      src_tokens = self._tokenizer.tokenize(
          element["question"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["answer"], add_eos=True
      )
    elif "instruction" in element.keys() and "response" in element.keys():  ## Instruction datasets
      if "input" in element.keys() and element["input"]: # Ensure 'input' exists and is not empty
        full_instruction = f"{element['instruction']}\n{element['input']}"
      else:
        full_instruction = element["instruction"]
        src_tokens = self._tokenizer.tokenize(
          full_instruction,
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
        )
      dst_tokens = self._tokenizer.tokenize(
          element["response"], add_eos=True
      )
    elif "text" in element.keys() and "summary" in element.keys():  ## Summarization datasets
      src_tokens = self._tokenizer.tokenize(
          element["text"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["summary"], add_eos=True
      )
    elif "instruction" in element.keys() and "output" in element.keys():  ## Medical instruction datasets
      src_tokens = self._tokenizer.tokenize(
          element["instruction"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["output"], add_eos=True
      )
    elif "question" in element.keys() and "answer" in element.keys():  ## Medical QA datasets
      src_tokens = self._tokenizer.tokenize(
          element["question"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["answer"], add_eos=True
      )
    elif "context" in element.keys() and "question" in element.keys() and "answer" in element.keys():  ## Medical context QA
      # Combine context and question
      combined_input = f"Context: {element['context']}\nQuestion: {element['question']}"
      src_tokens = self._tokenizer.tokenize(
          combined_input,
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["answer"], add_eos=True
      )
    else:
      raise ValueError(f"Unsupported dataset format: {element.keys()}")
    return src_tokens, dst_tokens


class _BuildTrainInput(grain.MapTransform):
  """Build a TrainingInput from a tuple of source and destination tokens."""

  def __init__(self, max_seq_len: int, pad_value: int | bool):
    self._max_seq_len = max_seq_len
    self._pad_value = pad_value

  def map(self, tokens: tuple[np.ndarray, np.ndarray]) -> TrainingInput:
    src_tokens, dst_tokens = tokens

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = np.concat([src_tokens, dst_tokens], axis=0)

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    q_mask = np.zeros_like(src_tokens, dtype=np.bool)
    a_mask = np.ones_like(dst_tokens, dtype=np.bool)
    mask = np.concat([q_mask, a_mask], axis=0)

    # If the input tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._pad_value)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._pad_up_to_max_len(mask, 0)

    return TrainingInput(input_tokens=tokens, input_mask=mask)

  def _pad_up_to_max_len(
      self, input_tensor: np.ndarray, pad_value: int
  ) -> np.ndarray:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self._max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode="constant",
        constant_values=pad_value,
    )


class _FilterOverlength(grain.FilterTransform):
  """Filter out overlength examples."""

  def __init__(self, max_seq_len: int):
    self._max_seq_len = max_seq_len

  def filter(self, element: TrainingInput) -> bool:
    return element.input_tokens.shape[0] <= self._max_seq_len
