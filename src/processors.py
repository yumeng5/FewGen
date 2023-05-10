# Adapted from https://github.com/princeton-nlp/LM-BFF/blob/main/src/processors.py

import os
import logging
import numpy as np
import json
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import logging
import dataclasses
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OurInputExample(InputExample):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    true: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class GenDataProcessor(DataProcessor):
    """Processor for generated data set"""

    def _read_json(self, file_dir):
        data_dict = json.load(open(file_dir, 'r'))
        return data_dict

    def _create_examples_from_json(self, data_dict, set_type="train"):
        examples = []
        for (i, data) in enumerate(data_dict):
            guid = "%s-%s" % (set_type, i)
            if "text1" in data:
                text_a = data["text1"]
                text_b = data["text2"]
            else:
                text_a = data["text"]
                text_b = None
            label = data["label"]
            if "true" in data:
                true = data["true"]
            else:
                true = None
            examples.append(OurInputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, true=true))
        return examples


class MrpcProcessor(GenDataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_gen_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(self._read_json(os.path.join(data_dir, "gen-train.json")))

    def get_train_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", select_label)

    def get_dev_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", select_label)
    
    def get_test_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", select_label)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, select_label=None):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            if select_label is None or select_label == label:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(GenDataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_gen_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(self._read_json(os.path.join(data_dir, "gen-train.json")))

    def get_train_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", select_label)

    def get_dev_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev", select_label)

    def get_test_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched", select_label)

    def get_labels(self):
        """See base class."""
        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type, select_label=None):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if select_label is None or select_label == label:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_test_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched", select_label)


class ColaProcessor(GenDataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_gen_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(self._read_json(os.path.join(data_dir, "gen-train.json")))

    def get_train_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", select_label)

    def get_dev_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", select_label)
    
    def get_test_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", select_label)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, select_label=None):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        text_index = 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            if select_label is None or select_label == label:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(GenDataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_gen_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(self._read_json(os.path.join(data_dir, "gen-train.json")))

    def get_train_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", select_label)

    def get_dev_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", select_label)
    
    def get_test_examples(self, data_dir, select_label=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", select_label)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, select_label=None):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            if select_label is None or select_label == label:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QqpProcessor(GenDataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_gen_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(self._read_json(os.path.join(data_dir, "gen-train.json")))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 3
        q2_index = 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(GenDataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_gen_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(self._read_json(os.path.join(data_dir, "gen-train.json")))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(GenDataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_gen_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(self._read_json(os.path.join(data_dir, "gen-train.json")))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


processors_mapping = {
    "cola": ColaProcessor(),
    "mnli": MnliProcessor(),
    "mnli-mm": MnliMismatchedProcessor(),
    "mrpc": MrpcProcessor(),
    "sst-2": Sst2Processor(),
    "qqp": QqpProcessor(),
    "qnli": QnliProcessor(),
    "rte": RteProcessor(),
}

num_labels_mapping = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
}

output_modes_mapping = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "cola": glue_compute_metrics,
    "mnli": glue_compute_metrics,
    "mnli-mm": glue_compute_metrics,
    "mrpc": glue_compute_metrics,
    "sst-2": glue_compute_metrics,
    "qqp": glue_compute_metrics,
    "qnli": glue_compute_metrics,
    "rte": glue_compute_metrics,
}

# Single-sequence or sequence-pair task
task_type_mapping = {
    "mnli": "pair",
    "qqp": "pair",
    "qnli": "pair",
    "sst-2": "single",
    "cola": "single",
    "rte": "pair",
    "mrpc": "pair",
}

# Control code used by CTRL as the starting token
control_code_mapping = {
    "mnli": "Wikipedia",
    "qqp": "Links",
    "qnli": "Links",
    "sst-2": {"0": "Reviews Rating: 1.0", "1": "Reviews Rating: 5.0"},
    "cola": "Links",
    "rte": "Wikipedia",
    "mrpc": "Wikipedia",
}


# Valid stop tokens used to terminate a sequence
stop_tokens_mapping = {
    "mnli": ['. '],
    "qqp": ['? ', '?\n'],
    "qnli": ['. '],
    "sst-2": ['. ', '? ', '! ', '\n'],
    "cola": ['. ', '? ', '! '],
    "rte": ['. '],
    "mrpc": ['. '],
}

# Initialization prompts for generator tuning 
prompt_mapping = {
    "mnli": {
        "entailment": ["Sentence 1 implies Sentence 2. Sentence 1:", "Sentence 2:"],
        "neutral": ["Sentence 2 supplements Sentence 1. Sentence 1:", "Sentence 2:"],
        "contradiction": ["Sentence 2 contradicts Sentence 1. Sentence 1:", "Sentence 2:"],
    },
    "qqp": {
        "0": ["Question 1 is equivalent to Question 2. Question 1:", "Question 2:"],
        "1": ["Question 1 is different from Question 2. Question 1:", "Question 2:"]
    },
    "qnli": {
        "entailment": ["Paragraph is relevant to Question. Question:", "Paragraph:"],
        "not_entailment": ["Paragraph is irrelevant to Question. Question:", "Paragraph:"],
    },
    "sst-2": {
        "0": "negative movie review:",
        "1": "positive movie review:",
    },
    "cola": {
        "0": "Linguistically incorrect sentence:",
        "1": "Linguistically correct sentence:",
    },
    "mrpc": {
        "1": ["Sentence 1 is equivalent to Sentence 2. Sentence 1:", "Sentence 2:"], 
        "0": ["Sentence 1 is different from Sentence 2. Sentence 1:", "Sentence 2:"]
    },
    "rte": {
        "entailment": ["Sentence 1 implies Sentence 2. Sentence 1:", "Sentence 2:"], 
        "not_entailment": ["Sentence 2 supplements Sentence 1. Sentence 1:", "Sentence 2:"]
    },
}
