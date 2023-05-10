# Largely adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import CTRLLMHeadModel
from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_utils import (
    GreedySearchEncoderDecoderOutput, 
    GreedySearchDecoderOnlyOutput,
    SampleEncoderDecoderOutput, 
    SampleDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput, 
    BeamSearchDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput, 
    BeamSampleDecoderOnlyOutput
)
from transformers.generation_logits_process import LogitsProcessorList

from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

@dataclass
class CausalLMOutputWithPastAndProb(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    log_probs: Optional[torch.FloatTensor] = None
    disc_loss: Optional[torch.FloatTensor] = None


GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]


class LinearBuf(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class PrefixCTRL(CTRLLMHeadModel):

    def __init__(self, config, task=None, label_list=None, prefix_len=10, control_code_len=1, infix_len=10, default_mode='full', bos_id=-1, eos_id=-1, meta_weight=True):
        r"""
        prefix_len: length of prefix vectors for optimization
        control_code_len: length of control code for CTRL
        infix_len: length of infix tokens (used for sentence-pair tasks)
        meta_weight: train with meta weighting
        """
        super().__init__(config)
        if hasattr(config, 'default_mode'):
            default_mode = config.default_mode
        if hasattr(config, 'task'):
            task = config.task
        else:
            assert task is not None, "task is None!"
        if hasattr(config, 'label_list'):
            label_list = config.label_list
        else:
            assert label_list is not None, "label_list is None!"
        if hasattr(config, 'prefix_len'):
            prefix_len = config.prefix_len
        if hasattr(config, 'control_code_len'):
            control_code_len = config.control_code_len
        if hasattr(config, 'infix_len'):
            infix_len = config.infix_len
        self.default_mode = default_mode
        self.control_code_len = control_code_len
        self.config = config
        self.config.default_mode = default_mode
        self.config.task = task
        self.config.label_list = label_list
        self.config.meta_weight = meta_weight
        self.dropout = nn.Dropout(config.embd_pdrop)
        # setup tunable embedding parameters for newly added special tokens
        if default_mode != 'full':
            self.bos_id = bos_id
            self.eos_id = eos_id
            # BOS and EOS token embeddings
            self.special_emb = nn.Parameter(torch.randn(2, config.n_embd))
            self.special_lm_head = nn.Linear(config.n_embd, 2, bias=True)
            # tie weights
            self.special_lm_head.weight = self.special_emb
        if 'prefix' in default_mode:
            self.prefix_len = prefix_len
            self.prefix_params = nn.Parameter(torch.randn(len(label_list), config.n_layer*2, 1, config.n_head, self.prefix_len, config.n_embd//config.n_head))
            # Control code embeddings will not be updated, thus separated from prefix parameters
            self.control_params = nn.Parameter(torch.randn(len(label_list), config.n_layer*2, 1, config.n_head, self.control_code_len, config.n_embd//config.n_head))
            self.config.prefix_len = self.prefix_len
        if 'infix' in default_mode:
            self.infix_len = infix_len
            self.infix_emb = nn.Parameter(torch.randn(len(label_list), 1, self.infix_len, config.n_embd))
            self.config.infix_len = self.infix_len
    
    def init_buffers(self):
        self.special_emb_buf = self.special_emb.data.clone().requires_grad_(True)
        self.prefix_params_buf = self.prefix_params.data.clone().requires_grad_(True)
        weight = self.special_emb_buf
        bias = self.special_lm_head.bias.data.clone().requires_grad_(True)
        self.special_lm_head_buf = LinearBuf(weight, bias)
        if hasattr(self, "infix_emb"):
            self.infix_emb_buf = self.infix_emb.data.clone().requires_grad_(True)

    def named_meta_params(self):
        meta_names = ["special_emb_buf", "prefix_params_buf", "infix_emb_buf", "special_lm_head_buf.bias"]
        all_meta_params = []
        for name in meta_names:
            if name == "special_lm_head_buf.bias" and hasattr(self, "special_lm_head_buf"):
                all_meta_params.append((name, self.special_lm_head_buf.bias))
                continue
            if hasattr(self, name):
                all_meta_params.append((name, getattr(self, name)))
        for n, p in all_meta_params:
            yield n, p

    def meta_params(self):
        for n, p in self.named_meta_params():
            yield p

    def init_prefix_param(self, prefix_params, control_code_len):
        assert prefix_params.shape[0] == len(self.config.label_list)
        self.control_params = nn.Parameter(prefix_params[..., :control_code_len, :])
        if prefix_params.shape[-2] > 1: # Initial prompts are provided
            self.prefix_params = nn.Parameter(prefix_params[..., control_code_len:, :])
        self.config.prefix_len = self.prefix_params.shape[-2]
        self.config.control_code_len = self.control_params.shape[-2]

    def init_infix_param(self, input_ids):
        self.infix_emb = nn.Parameter(self.get_input_embeddings()(input_ids))
        self.infix_len = self.infix_emb.shape[-2]
        self.config.infix_len = self.infix_len

    def freeze_unoptimized_params(self):
        # Freeze all model parameters not subject to optimization
        param_set = {'special_emb', 'special_lm_head'}
        if 'prefix' in self.config.default_mode: 
            param_set.add('prefix_params')
        if 'infix' in self.config.default_mode: 
            param_set.add('infix_emb')
        for name, param in self.named_parameters():
            if name not in param_set:
                param.requires_grad = False
    
    def update_params(self, lr, grad):
        for tgt, src in zip(filter(lambda p: p[1].requires_grad, self.named_meta_params()), grad):
            name, param = tgt
            new_param = param - lr * src
            self.set_params(self, name, new_param)

    def set_params(self, curr_module, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_module.named_children():
                if module_name == name:
                    self.set_params(mod, rest, param)
                    break
        else:
            setattr(curr_module, name, param)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        cat_label=None,
        infix_pos=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mode=None,
        require_prob=False,
        meta_model=None,
        meta_lr=None,
        weight_net=None,
        weight_net_optimizer=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        if labels is not None:
            labels = labels.to(input_ids)
        if mode is None:
            mode = self.default_mode
        if mode == 'full' or past_key_values is not None:
            return super().forward(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, 
                                   token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, 
                                   inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, 
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        elif 'prefix' in mode:
            disc_loss = None
            if not self.config.meta_weight:
                meta_model = None
            if meta_model is not None:
                meta_model.load_state_dict(self.state_dict())
                meta_model.init_buffers()
                # print(f"\n\n ### Trainable params: {[n for n, p in meta_model.named_meta_params() if p.requires_grad]} ###")
                outputs = meta_model.meta_forward(input_ids=input_ids, 
                                                  past_key_values=past_key_values, 
                                                  attention_mask=attention_mask, 
                                                  token_type_ids=token_type_ids, 
                                                  cat_label=cat_label,
                                                  infix_pos=infix_pos,
                                                  position_ids=position_ids,
                                                  head_mask=head_mask, 
                                                  inputs_embeds=inputs_embeds, 
                                                  labels=labels, 
                                                  use_cache=use_cache, 
                                                  output_attentions=output_attentions, 
                                                  output_hidden_states=output_hidden_states,
                                                  return_dict=return_dict,
                                                  mode=mode,
                                                  weight_net=weight_net,)
                meta_model.zero_grad()
                grads = torch.autograd.grad(outputs.loss, (meta_model.meta_params()), create_graph=True)
                meta_model.update_params(meta_lr, grads)
                del grads
                outputs = meta_model.meta_forward(input_ids=input_ids, 
                                                  past_key_values=past_key_values, 
                                                  attention_mask=attention_mask, 
                                                  token_type_ids=token_type_ids, 
                                                  cat_label=cat_label,
                                                  infix_pos=infix_pos,
                                                  position_ids=position_ids,
                                                  head_mask=head_mask, 
                                                  inputs_embeds=inputs_embeds, 
                                                  labels=labels, 
                                                  use_cache=use_cache, 
                                                  output_attentions=output_attentions, 
                                                  output_hidden_states=output_hidden_states,
                                                  return_dict=return_dict,
                                                  mode=mode,
                                                  weight_net=weight_net,
                                                  compute_all_label=True,)
                weight_net_optimizer.zero_grad()
                disc_loss = outputs.loss
                disc_loss.backward()
                weight_net_optimizer.step()

            # For inference
            if weight_net is None:
                if self.control_params is not None:
                    prefix_params = torch.cat((self.control_params, self.prefix_params), dim=-2)
                else:
                    prefix_params = self.prefix_params
                if attention_mask is not None:
                    prefix_mask = torch.ones((attention_mask.shape[0], prefix_params.shape[-2])).to(attention_mask)
                    attention_mask = torch.cat((prefix_mask, attention_mask), dim=-1)
                else:
                    attention_mask = None
                bsz = input_ids.shape[0]
                prefix = prefix_params.expand(-1, -1, bsz, -1, -1, -1)
                prefix = self.dropout(prefix)
                cat_label_prefix = cat_label.view(1, 1, bsz, 1, 1, 1).expand(-1, prefix.shape[1], -1, prefix.shape[3], prefix.shape[4], prefix.shape[5])
                prefix = prefix.gather(0, cat_label_prefix).squeeze(0)
                past_key_values = prefix.split(2)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                inputs_embeds[input_ids == self.bos_id] = self.special_emb[0]
                inputs_embeds[input_ids == self.eos_id] = self.special_emb[1]

                if "infix" in mode:
                    infix_valid = infix_pos.sum(dim=-1) == self.infix_len
                    cat_label_infix = cat_label[infix_valid]
                    num_infix = len(cat_label_infix)
                    infix_embeds = self.infix_emb.expand(-1, num_infix, -1, -1)
                    cat_label_infix = cat_label_infix.view(1, num_infix, 1, 1).expand(-1, -1, infix_embeds.shape[-2], infix_embeds.shape[-1])
                    infix_embeds = infix_embeds.gather(0, cat_label_infix).squeeze(0)
                    infix_embeds = infix_embeds.reshape(-1, self.infix_emb.shape[-1])
                    inputs_embeds[infix_pos == 1] = infix_embeds
            # For training
            else:
                if self.control_params is not None:
                    prefix_params = torch.cat((self.control_params, self.prefix_params), dim=-2)
                else:
                    prefix_params = self.prefix_params
                if attention_mask is not None:
                    prefix_mask = torch.ones((attention_mask.shape[0], prefix_params.shape[-2])).to(attention_mask)
                    attention_mask = torch.cat((prefix_mask, attention_mask), dim=-1)
                    attention_mask = attention_mask.repeat(self.prefix_params.shape[0], 1)
                else:
                    attention_mask = None
                bsz = input_ids.shape[0]
                prefix = prefix_params.expand(-1, -1, bsz, -1, -1, -1)
                prefix = prefix.permute((1, 0, 2, 3, 4, 5))
                prefix = prefix.reshape(prefix.shape[0], prefix.shape[1]*prefix.shape[2], prefix.shape[3], prefix.shape[4], prefix.shape[5])
                prefix = self.dropout(prefix)
                
                past_key_values = prefix.split(2)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                inputs_embeds[input_ids == self.bos_id] = self.special_emb[0]
                inputs_embeds[input_ids == self.eos_id] = self.special_emb[1]
                inputs_embeds = inputs_embeds.repeat(self.prefix_params.shape[0], 1, 1)

                if "infix" in mode:
                    infix_valid = infix_pos.sum(dim=-1) == self.infix_len
                    num_infix = infix_valid.sum()
                    infix_embeds = self.infix_emb.expand(-1, num_infix, -1, -1)
                    infix_embeds = infix_embeds.reshape(-1, self.infix_emb.shape[-1])
                    inputs_embeds[infix_pos.repeat(self.prefix_params.shape[0], 1) == 1] = infix_embeds
            
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            transformer_outputs = self.transformer(
                None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            past_key_values = transformer_outputs[1]
            lm_logits = self.lm_head(hidden_states)
            special_lm_logits = self.special_lm_head(hidden_states)
            lm_logits[..., self.bos_id] = special_lm_logits[..., 0]
            lm_logits[..., self.eos_id] = special_lm_logits[..., 1]
            
            loss = None
            if labels is not None:
                # For inference
                if weight_net is None:
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(reduction='mean')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    lm_logits = lm_logits.view(self.prefix_params.shape[0], bsz, -1, lm_logits.shape[-1])
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    expand_labels = labels.repeat(self.prefix_params.shape[0], 1)
                    shift_labels = expand_labels.view(self.prefix_params.shape[0], bsz, -1)[..., 1:].contiguous()
                    shift_logits = shift_logits[shift_labels != -100]
                    shift_labels = shift_labels[shift_labels != -100]
                    token_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    expand_cat_label = cat_label.view(bsz, 1).expand(-1, lm_logits.shape[-2])[..., 1:].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    expand_cat_label = expand_cat_label[shift_labels != -100]
                    token_logits = token_logits.view(self.prefix_params.shape[0], -1).permute(1, 0)
                    loss_fct = CrossEntropyLoss(reduction='none')
                    disc_loss = loss_fct(token_logits, expand_cat_label.view(-1))
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    cat_label = cat_label.view(1, bsz, 1, 1).expand(-1, -1, shift_logits.shape[-2], shift_logits.shape[-1])
                    shift_logits = shift_logits.gather(0, cat_label).squeeze(0)
                    with torch.no_grad():
                        weights = weight_net(disc_loss.view(-1, 1)).squeeze(-1)
                    disc_loss = disc_loss.mean()
                    weights = F.softmax(weights, dim=-1)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = (loss[shift_labels.view(-1) != -100] * weights).sum()

            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            log_probs = None
            if require_prob and labels is not None:
                valid_pos = shift_labels != -100
                full_log_probs = torch.zeros_like(shift_labels, dtype=lm_logits.dtype)
                shift_labels = shift_labels[valid_pos]
                log_probs = F.log_softmax(shift_logits[valid_pos], dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1))
                full_log_probs[valid_pos] = token_log_probs.squeeze(-1)
                log_probs = full_log_probs.sum(dim=-1) / valid_pos.sum(dim=-1)

            return CausalLMOutputWithPastAndProb(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                log_probs=log_probs,
                disc_loss=disc_loss,
            )
            
    def meta_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        cat_label=None,
        infix_pos=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mode=None,
        require_prob=False,
        weight_net=None,
        compute_all_label=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        if labels is not None:
            labels = labels.to(input_ids)
        if mode is None:
            mode = self.default_mode
        if mode == 'full' or past_key_values is not None:
            return super().forward(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, 
                                   token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, 
                                   inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, 
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        elif 'prefix' in mode:
            if self.control_params is not None:
                prefix_params = torch.cat((self.control_params, self.prefix_params_buf), dim=-2)
            else:
                prefix_params = self.prefix_params_buf
            if attention_mask is not None:
                prefix_mask = torch.ones((attention_mask.shape[0], prefix_params.shape[-2])).to(attention_mask)
                attention_mask = torch.cat((prefix_mask, attention_mask), dim=-1)
                attention_mask = attention_mask.repeat(self.prefix_params_buf.shape[0], 1)
            else:
                attention_mask = None
            bsz = input_ids.shape[0]
            prefix = prefix_params.expand(-1, -1, bsz, -1, -1, -1)
            prefix = prefix.permute((1, 0, 2, 3, 4, 5))
            prefix = prefix.reshape(prefix.shape[0], prefix.shape[1]*prefix.shape[2], prefix.shape[3], prefix.shape[4], prefix.shape[5])
            prefix = self.dropout(prefix)
            
            past_key_values = prefix.split(2)
            inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[input_ids == self.bos_id] = self.special_emb_buf[0]
            inputs_embeds[input_ids == self.eos_id] = self.special_emb_buf[1]
            inputs_embeds = inputs_embeds.repeat(self.prefix_params_buf.shape[0], 1, 1)

            if "infix" in mode:
                infix_valid = infix_pos.sum(dim=-1) == self.infix_len
                num_infix = infix_valid.sum()
                infix_embeds = self.infix_emb_buf.expand(-1, num_infix, -1, -1)
                infix_embeds = infix_embeds.reshape(-1, self.infix_emb_buf.shape[-1])
                inputs_embeds[infix_pos.repeat(self.prefix_params_buf.shape[0], 1) == 1] = infix_embeds
            
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            transformer_outputs = self.transformer(
                None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            past_key_values = transformer_outputs[1]
            lm_logits = self.lm_head(hidden_states)
            special_lm_logits = self.special_lm_head_buf(hidden_states)
            lm_logits[..., self.bos_id] = special_lm_logits[..., 0]
            lm_logits[..., self.eos_id] = special_lm_logits[..., 1]
            lm_logits = lm_logits.view(self.prefix_params.shape[0], bsz, -1, lm_logits.shape[-1])
            shift_logits = lm_logits[..., :-1, :].contiguous()
            expand_labels = labels.repeat(self.prefix_params.shape[0], 1)
            shift_labels = expand_labels.view(self.prefix_params.shape[0], bsz, -1)[..., 1:].contiguous()
            shift_logits = shift_logits[shift_labels != -100]
            shift_labels = shift_labels[shift_labels != -100]
            token_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            expand_cat_label = cat_label.view(bsz, 1).expand(-1, lm_logits.shape[-2])[..., 1:].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            expand_cat_label = expand_cat_label[shift_labels != -100]
            token_logits = token_logits.view(self.prefix_params.shape[0], -1).permute(1, 0)
            loss_fct = CrossEntropyLoss(reduction='none')
            disc_loss = loss_fct(token_logits, expand_cat_label.view(-1))
            if compute_all_label:
                loss = disc_loss.mean()
            else:
                weights = weight_net(disc_loss.view(-1, 1).detach()).squeeze(-1)
                weights = F.softmax(weights, dim=-1)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                cat_label = cat_label.view(1, bsz, 1, 1).expand(-1, -1, shift_logits.shape[-2], shift_logits.shape[-1])
                shift_logits = shift_logits.gather(0, cat_label).squeeze(0)
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = (loss[shift_labels.view(-1) != -100] * weights).sum()

            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            log_probs = None
            if require_prob and labels is not None:
                valid_pos = shift_labels != -100
                full_log_probs = torch.zeros_like(shift_labels, dtype=lm_logits.dtype)
                shift_labels = shift_labels[valid_pos]
                log_probs = F.log_softmax(shift_logits[valid_pos], dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1))
                full_log_probs[valid_pos] = token_log_probs.squeeze(-1)
                log_probs = full_log_probs.sum(dim=-1) / valid_pos.sum(dim=-1)

            return CausalLMOutputWithPastAndProb(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                log_probs=log_probs,
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        infix_pos: Optional[torch.LongTensor] = None,
        cat_label: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it with
                :obj:`bos_token_id` and a batch size of 1.
            max_length (:obj:`int`, `optional`, defaults to :obj:`model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (:obj:`int`, `optional`, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                :obj:`max_new_tokens` or :obj:`max_length` but not both, they serve the same purpose.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
                ``decoder_input_ids``.
            bad_words_ids(:obj:`List[List[int]]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(:obj:`float`, `optional`, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
                <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID :obj:`batch_id` and
                :obj:`input_ids`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the batch ID :obj:`batch_id` and the previously generated tokens :obj:`inputs_ids`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            forced_bos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
                Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
                needs to be the target language token.
            forced_eos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the last generated token when :obj:`max_length` is reached.
            remove_invalid_values (:obj:`bool`, `optional`):
                Whether to remove possible `nan` and `inf` outputs of the model to prevent the generation method to
                crash. Note that using ``remove_invalid_values`` can slow down generation.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
                model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with `decoder_`.

        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.LongTensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`torch.FloatTensor`.

                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`

                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.SampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput`

        Examples::
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> # do greedy decoding without providing a prompt
            >>> outputs = model.generate(max_length=40)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> document = (
            ... "at least two people were killed in a suspected bomb attack on a passenger bus "
            ... "in the strife-torn southern philippines on monday , the military said."
            ... )
            >>> # encode input context
            >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
            >>> # generate 3 independent sequences using beam search decoding (5 beams)
            >>> # with T5 encoder-decoder model conditioned on short news article.
            >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate 3 candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
            >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
            >>> # "Legal" is one of the control codes for ctrl
            >>> input_context = "Legal My neighbor is"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> input_context = "My cute dog"
            >>> # get tokens of words that should not be generated
            >>> bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate sequences without allowing bad_words to be generated
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        """
        # set init values
        if max_length is None and max_new_tokens is None:
            # Both are None, default
            max_length = self.config.max_length
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set but they serve the same purpose.", UserWarning
            )

        max_length = max_length if max_length is not None else self.config.max_length
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        assert input_ids is not None, "no starting tokens provided for generation!"

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )

        cur_len = input_ids.shape[-1]
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, max_new_tokens=max_new_tokens, start_length=cur_len
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                infix_pos,
                cat_label,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # sample
            return self.sample(
                input_ids,
                infix_pos,
                cat_label,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            batch_size = input_ids.shape[0] * num_return_sequences

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # interleave with `num_beams * num_return_sequences`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            diverse_beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        infix_pos: Optional[torch.LongTensor] = None,
        cat_label: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.

            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the :obj:`forward` function of the
                model. If model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ... AutoTokenizer,
            ... AutoModelForCausalLM,
            ... LogitsProcessorList,
            ... MinLengthLogitsProcessor,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                infix_pos=infix_pos,
                cat_label=cat_label,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
    
    def sample(
        self,
        input_ids: torch.LongTensor,
        infix_pos: Optional[torch.LongTensor] = None,
        cat_label: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.
            logits_warper (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
                modeling head applied before multinomial sampling at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForCausalLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    TopKLogitsWarper,
            ...    TemperatureLogitsWarper,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                infix_pos=infix_pos,
                cat_label=cat_label,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
    