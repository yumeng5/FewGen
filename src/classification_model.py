# Adapted from https://github.com/princeton-nlp/LM-BFF/blob/main/src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLMHead, RobertaClassificationHead

import logging
logger = logging.getLogger(__name__)


class ConsistLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        loss_fct = nn.KLDivLoss(reduction='batchmean')
        loss = loss_fct(log_probs, targets)
        return loss


class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.label_word_list = None
        self.temporal_ensemble = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        ensemble_label=None,
    ):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        loss = None
        consist_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.model_args.smooth)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            if self.temporal_ensemble:
                loss_fct = ConsistLoss()
                consist_loss = loss_fct(logits.view(-1, logits.size(-1)), ensemble_label.to(logits))
                loss += self.model_args.reg_weight * consist_loss
        output = (logits,)
        return ((loss,) + output) if loss is not None else output


class RobertaForSequenceClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.temporal_ensemble = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        ensemble_label=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        logits = self.classifier(outputs[0])

        loss = None
        consist_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.model_args.smooth)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            if self.temporal_ensemble:
                loss_fct = ConsistLoss()
                consist_loss = loss_fct(logits.view(-1, logits.size(-1)), ensemble_label.to(logits))
                loss += self.model_args.reg_weight * consist_loss
        output = (logits,)
        return ((loss,) + output) if loss is not None else output
