#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
logging.basicConfig(level = logging.INFO)
import math
import os
import sys
from typing import Optional
from dataclasses import dataclass, field

import transformers
from src.dataset import GenerationDataset
from transformers import GlueDataTrainingArguments

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from src.generation_trainer import GenTrainer as Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from src.generation_model import PrefixCTRL
import torch
from src.processors import control_code_mapping, prompt_mapping, task_type_mapping, processors_mapping

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    train_mode: str = field(
        default="prefix-infix",
        metadata={"help": "Generator training mode"},
    )
    meta_weight: bool = field(
        default=False, metadata={"help": "Train model with meta weighting"}
    )
    prefix_len: int = field(
        default=10,
        metadata={"help": "Prompt length for prefix tuning"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments(GlueDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    gen_label: str = field(default=None, metadata={"help": "The label of the generated texts."})
    task_name: str = field(
        default=None,
        metadata={"help": "Task name"}
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "Path to dataset"}
    )
    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )
    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    no_save: bool = field(
        default=False, metadata={"help": "Do not save trained model"}
    )
    eval_gen_file: str = field(
        default=None,
        metadata={"help": "Path to generated json file to be evaluated"}
    )

@dataclass
class DynamicTrainingArguments(TrainingArguments):
    weight_net_lr: float = field(
        default=1e-3,
        metadata={"help": "learning rate of weight net"}
    )
    meta_lr: float = field(
        default=1e-2,
        metadata={"help": "learning rate of meta model"}
    )
    weight_net_decay: float = field(
        default=1e-4,
        metadata={"help": "weight decay of weight net"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    task_type = task_type_mapping[data_args.task_name]
    if 'infix' in model_args.train_mode and task_type == 'single':
        model_args.train_mode = model_args.train_mode.replace("-infix", "")

    # use the pretrained CTRL model
    if model_args.model_name_or_path == 'ctrl':
        label_list = list(prompt_mapping[data_args.task_name].keys())
        model = PrefixCTRL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                task=data_args.task_name,
                label_list=label_list,
                prefix_len=model_args.prefix_len,
                default_mode=model_args.train_mode,
                bos_id=tokenizer.bos_token_id,
                eos_id=tokenizer.eos_token_id,
                meta_weight=model_args.meta_weight
            )
    # load a tuned CTRL model saved in local paths
    else:
        model = PrefixCTRL.from_pretrained(
                model_args.model_name_or_path,
            )
        label_list = model.config.label_list
        data_args.task_name = model.config.task
        model_args.train_mode = model.config.default_mode

    model.resize_token_embeddings(len(tokenizer))
    
    control_code = control_code_mapping[data_args.task_name]
    try:
        task_prompt = prompt_mapping[data_args.task_name]
        prefix_init = []
        infix_init = []
        for label in label_list:
            prompt = task_prompt[label]
            if type(prompt) != list:
                prompt = [prompt]
            if task_type == "pair" and len(prompt) == 1:
                prompt = [None] + prompt
            prefix_init.append(prompt[0])
            if task_type == "pair":
                infix_init.append(prompt[1])
            else:
                infix_init.append(None)
    except:
        prompt = None
        prefix_init = None
        infix_init = None

    # Initialize prefix tuning parameters
    if prefix_init is not None:
        all_init_key_values = []
        control_code_len = -1
        for i, init in enumerate(prefix_init):
            print('Initialization:', init)
            label_control_code = control_code[label_list[i]] if type(control_code) == dict else control_code
            print(f"Control code: {label_control_code}")
            control_code_input = tokenizer([label_control_code])
            if control_code_len == -1:
                control_code_len = len(control_code_input['input_ids'][0])
            else:
                assert control_code_len == len(control_code_input['input_ids'][0]), "Control code length is not consistent across labels!"
            init_input = tokenizer([label_control_code + ' ' + init])
            output = model(input_ids=torch.tensor(init_input['input_ids']), mode="full")
            print(f"Initialized prefix prompt length: {len(init_input['input_ids'][0])}")
            past_key_values = output.past_key_values
            init_key_values = torch.cat(past_key_values, dim=0).unsqueeze(0)
            # print(init_key_values.shape)
            all_init_key_values.append(init_key_values)
        if training_args.do_train:
            model.init_prefix_param(torch.cat(all_init_key_values, dim=0), control_code_len)

    model.freeze_unoptimized_params() 
    control_code = None

    if 'infix' in model_args.train_mode and infix_init is not None:
        all_init_infix = []
        for i, init in enumerate(infix_init):
            init_input = tokenizer([init])
            print(f"Initialized infix prompt length: {len(init_input['input_ids'][0])}")
            all_init_infix.append(torch.tensor(init_input['input_ids']).unsqueeze(0))
        model.init_infix_param(torch.cat(all_init_infix, dim=0))
        for name, param in model.named_parameters():
            if "infix" in name:
                param.requires_grad = True

    if "no-prompt" in model_args.train_mode or task_type == "single":
        prompt = [None, None]
    else:
        prompt = [None, prompt[1]]

    return_infix = "infix" in model_args.train_mode
    train_dataset = (
        GenerationDataset(data_args, 
                          tokenizer=tokenizer, 
                          prompt=prompt, 
                          processor=processors_mapping[data_args.task_name], 
                          control_code=control_code, 
                          return_infix=return_infix, 
                          mode="train")
    )
    dev_dataset = (
        GenerationDataset(data_args, 
                          tokenizer=tokenizer, 
                          prompt=prompt, 
                          processor=processors_mapping[data_args.task_name], 
                          control_code=control_code, 
                          return_infix=return_infix, 
                          mode="dev")
        if training_args.do_eval
        else None
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )
    # Training
    print(f"\n\n ### Trainable params: {[n for n, p in model.named_parameters() if p.requires_grad]} ###")
    print(f"\n\n ### num of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)} ###")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # Assign learned special token embeddings to model token embeddings
        with torch.no_grad():
            # Update the word embeddings for BOS and EOS
            word_embeddings = trainer.model.transformer.get_input_embeddings()
            word_embeddings.weight[tokenizer.bos_token_id] = trainer.model.special_emb[0]
            word_embeddings.weight[tokenizer.eos_token_id] = trainer.model.special_emb[1]
            trainer.model.transformer.set_input_embeddings(word_embeddings)

        # Update the biases, if the model is CTRL
        if 'CTRL' in type(model).__name__:
            with torch.no_grad():
                trainer.model.lm_head.bias[tokenizer.bos_token_id] = trainer.model.special_lm_head.bias[0]
                trainer.model.lm_head.bias[tokenizer.eos_token_id] = trainer.model.special_lm_head.bias[1]
        if not data_args.no_save:
            trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()