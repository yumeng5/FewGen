# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-generation/run_generation.py

import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
import json
import os

from transformers import (
    CTRLTokenizer,
    AutoTokenizer,
)
from generation_model import PrefixCTRL
from processors import task_type_mapping, control_code_mapping, prompt_mapping

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Repetition reward/penalty parameters
repetition_mapping = {
    "mnli": {
        "entailment": 1.1,
        "neutral": 1.5,
        "contradiction": 1.1,
    },
    "qqp": {
        "0": 1.5,
        "1": 1.0,
    },
    "qnli": {
        "entailment": 1.0,
        "not_entailment": 1.5,
    },
    "sst-2": {
        "0": 1.1,
        "1": 1.1,
    },
    "cola": {
        "0": 1.1,
        "1": 1.1,
    },
    "rte": {
        "entailment": 1.0,
        "not_entailment": 1.5,
    },
    "mrpc": {
        "1": 1.0,
        "0": 1.5,
    },
}


# Generated sequences containing bad tokens will be discarded
bad_tokens_mapping = {
    "mnli": ['\n'],
    "qqp": ['\n'],
    "qnli": ['?', '\n'],
    "sst-2": ['\n'],
    "cola": ['"', '“', '”', '\n'],
    "rte": ['\n'],
    "mrpc": ['\n'],
}


# If specified, generation will start with one of the given options
fix_start_mapping = {
    "sst-2": ["a", "one", "the", "this", "that", "i", "you", "it", "what"],
    "cola": ['Such', 'Again', 'Until', 'Her', 'These', 'Where', 'She', 'The', 'We',
             'Both', 'Under', 'At', 'Of', 'Doing', "You're", 'More', 'Between', 'All',
             'While', 'As', 'Our', 'Just', 'Once', 'His', 'Other', 'Most', 'In', 'My', 'Ours',
             'Before', 'When', 'He', 'There', 'Here', 'So', 'Because', 'You', 'Over',
             'During', 'Above', 'They', 'To', 'For', 'But', 'Only', 'Those', 'Against',
             'Your', 'After', 'Now', 'An', 'Too', 'Same', 'Its', 'From', 'Being', 'With',
             'A', 'Their', 'Each', "She's", 'It', 'No', 'Then', "It's", "You've", 'Some', 
             'Few', 'This', 'If', 'By', 'I'],
}


class FewGenGenerator():

    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # self.tokenizer = CTRLTokenizer.from_pretrained(args.model_name_or_path)
        self.tokenizer.model_max_length = 512
        self.linebreak_idx = self.tokenizer.convert_tokens_to_ids('\n')
        self.model = PrefixCTRL.from_pretrained(args.model_name_or_path)
        self.default_mode = self.model.default_mode
        args.task = self.model.config.task
        args.label_list = self.model.config.label_list
        self.label_map = {label: i for i, label in enumerate(args.label_list)}
        print(f"task: {args.task}; label: {args.label}")
        self.model.to(args.device)
        if args.fp16:
            self.model.half()
        self.set_seed(args.seed)
        self.task_type = task_type_mapping[args.task]
        self.stop_token = self.tokenizer.eos_token
        self.control_code = control_code_mapping[args.task] if 'ctrl' in args.model_name_or_path else None
        self.prompt = prompt_mapping[args.task][args.label]
        self.repetition = repetition_mapping[args.task][args.label]
        self.bad_tokens = bad_tokens_mapping[args.task]
        self.fix_start = fix_start_mapping[args.task] if args.task in fix_start_mapping else None
        if self.task_type == "pair":
            assert args.temperature == 0
            assert args.pretrain_corpus_dir is not None
            f = open(args.pretrain_corpus_dir)
            texts = f.readlines()
            texts = [text.strip() for text in texts]
            self.sampled_texts = np.random.permutation(texts)
        else:
            assert args.temperature > 0
            self.sampled_texts = None
        self.prompt_list = self.prompt if type(self.prompt) == list else [self.prompt]
        if args.temperature == 0:
            self.temp = 1
            self.do_sample = False
        else:
            self.temp = args.temperature
            self.do_sample = True

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def prepare_input(self, prompt_text):
        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not any(encoded_prompt[0] == x for x in self.tokenizer.control_codes.values()):
            logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
        return prompt_text

    def generate_one(self, seed, label, sample_text=None):
        self.set_seed(seed)
        start = ''
        # always start with control codes (when generator is CTRL)
        if self.default_mode == 'full' and 'ctrl' in self.model.__name__.lower():
            start = self.control_code + ' '
        prompts = prompt_mapping[self.args.task]
        prompt_list = prompts[label]
        repetition_penalty = repetition_mapping[self.args.task][label]
        prompt = prompt_list
        if type(prompt) == list:
            assert len(prompt) == 2 and sample_text is not None
            start_prompt = prompt[0]
            conj_prompt = prompt[1]
            lowercase_sampled = False
        else:
            start_prompt = prompt if sample_text is None else None
            conj_prompt = None if sample_text is None else prompt
            lowercase_sampled = False
        
        if self.default_mode != 'full':
            start_prompt = None
        if 'no-prompt' in self.default_mode:
            conj_prompt = '[BOS]'

        # append start prompt if any
        if start_prompt is not None and len(start_prompt) > 0:
            start += start_prompt + ' '
        prompt_text = start

        # append sample text if any
        if sample_text is not None:
            orig_sample_text = sample_text
            if lowercase_sampled:
                sample_text = orig_sample_text[0].lower() + orig_sample_text[1:]
            else:
                sample_text = orig_sample_text
            start += sample_text + ' '
        
        # append conjunction prompt if any
        infix_pos = []
        if conj_prompt is not None and len(conj_prompt) > 0:
            start_after = start + conj_prompt + ' '
            if 'infix' in self.default_mode:
                encoded_prompt_before = self.tokenizer.encode(
                    start, add_special_tokens=False, return_tensors="pt",
                )
                infix_pos += [0] * len(encoded_prompt_before[0])
                encoded_prompt_after = self.tokenizer.encode(
                    start_after, add_special_tokens=False, return_tensors="pt",
                )
                infix_pos += [1] * (len(encoded_prompt_after[0]) - len(encoded_prompt_before[0]))
            start = start_after
        
        # append fixed start tokens if any
        if self.fix_start is not None:
            choice_idx = np.random.choice(len(self.fix_start), 1)
            start_words = self.fix_start[choice_idx[0]]
            start += start_words + ' '
        else:
            start_words = None
        
        preprocessed_start_text = start
        encoded_start = self.tokenizer.encode(
            preprocessed_start_text, add_special_tokens=False, return_tensors="pt",
        )
        encoded_start = encoded_start.to(self.args.device)
        if encoded_start.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_start
            if 'infix' in self.default_mode:
                infix_pos += [0] * (len(input_ids[0]) - len(infix_pos))
                infix_pos = torch.tensor(infix_pos).unsqueeze(0).to(input_ids)
            else:
                infix_pos = None
        
        if sample_text is not None:
            max_len = len(input_ids[0]) + self.args.max_len
        else:
            max_len = self.args.max_len
        
        cat_label = torch.tensor([self.label_map[label]]).to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            infix_pos=infix_pos,
            cat_label=cat_label,
            max_length=max_len,
            temperature=self.temp,
            top_k=self.args.k,
            top_p=self.args.p,
            repetition_penalty=repetition_penalty,
            do_sample=self.do_sample,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_sequences = outputs["sequences"][0]

        generated_sequence = output_sequences
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        start = self.tokenizer.decode(encoded_start[0], clean_up_tokenization_spaces=True)
        start_len = len(start)

        # Discard generations starting with line breaks
        if text[start_len:].startswith("\n") or text[start_len:].startswith(" \n"):
            return None
        
        final_stop_idx = text.find(self.stop_token)
        if final_stop_idx == -1:
            return None

        # Remove all text after the stop token
        trunc_text = text[:final_stop_idx]

        total_sequence = (trunc_text[start_len:])
        total_sequence = total_sequence.strip()
        for bad_token in self.bad_tokens:
            if bad_token in total_sequence:
                return None
        if len(total_sequence) == 0:
            return None

        if start_words is not None:
            gen_text = start_words + ' ' + total_sequence
        else:
            gen_text = total_sequence
        if sample_text is not None:
            res = {"text1": orig_sample_text, 
                   "text2": gen_text, 
                   "label": label,
                   }
            if self.args.print_res:
                print(res)
        else:
            res = {"text": gen_text, 
                   "label": label,
                   }
            if self.args.print_res:
                print(res)
        return res

    def save_res(self, gen_res):
        os.makedirs(self.args.save_dir, exist_ok=True)
        save_name = os.path.join(self.args.save_dir, f"{self.args.task}_{self.args.label}_{self.args.num_gen}")
        with open(f"{save_name}.json", 'w') as f:
            res = json.dumps(gen_res)
            f.write(res)
            f.close()
            print(f"Generated results saved to {save_name}.json")

    def generate_all(self, label):
        gen_res = []
        text_set = set()
        i = 0
        pbar = tqdm(total=self.args.num_gen)
        while len(gen_res) < self.args.num_gen:
            if self.sampled_texts is None:
                sample_text = None
            else:
                sample_text = self.sampled_texts[i]
            res = self.generate_one(i, label, sample_text=sample_text)
            if res is not None:
                text = res["text"] if "text" in res else res["text1"]
                # Only save non-repetitive generations
                if text not in text_set:
                    gen_res.append(res)
                    text_set.add(text)
                    pbar.update(1)
            i += 1
        pbar.close()
        self.save_res(gen_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_corpus_dir', default=None,)
    parser.add_argument('--task', default='mnli',)
    parser.add_argument('--label', default='entailment',)
    parser.add_argument('--model_type', default='ctrl',)
    parser.add_argument('--model_name_or_path', default='ctrl',)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--no_cuda', default=False,)
    parser.add_argument('--fp16', default=False,)
    parser.add_argument('--num_gen', default=10, type=int)
    parser.add_argument('--max_len', default=60, type=int)
    parser.add_argument('--save_dir', default='temp_gen')
    parser.add_argument('--print_res', action='store_true')
    args = parser.parse_args()
    print(args)
    args.task = args.task.lower()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    generator = FewGenGenerator(args)
    generator.generate_all(args.label)


if __name__ == "__main__":
    main()
    