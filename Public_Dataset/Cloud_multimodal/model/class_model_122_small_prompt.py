"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict

import torch
from torch import nn
from .model import UniterPreTrainedModel
from .model_prompt import UniterModelPrompt
import json
from apex.normalization.fused_layer_norm import FusedLayerNorm
import logging
import copy
import numpy as np
import torch.nn.functional as F
logger = logging.getLogger(__name__)


class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = UniterConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UniterForClassSmallPrompt(torch.nn.Module):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config_file, img_dim, layer_num):
        super().__init__()
        self.config = UniterConfig.from_json_file(config_file)
        self.uniter = UniterModelPrompt(self.config, img_dim) 
        self.classifier = nn.Linear(self.config.hidden_size, 122)
        self.apply(self.init_weights)
        self.layer_num = layer_num

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        #new input
        img_feat = batch['img_feat'] #[b,2,l,d]
        userid = batch['userid']
        img_mask = batch['img_mask']
        txt_mask = batch['txt_mask']

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_mask, txt_mask, userid, #new input
                                      img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=True)
        sequence_output = sequence_output[self.layer_num-1]            
        pooled_output = self.uniter.pooler(sequence_output)
        outputs = self.classifier(pooled_output)
        return outputs

    def get_prompt(self, batch):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        img_feat = batch['img_feat'] #[b,2,l,d]
        userid = batch['userid']
        img_mask = batch['img_mask']
        txt_mask = batch['txt_mask']

        prompt = self.uniter.get_prompt(input_ids, position_ids,
                                      img_feat, img_mask, txt_mask, userid, 
                                      img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=True)
        return prompt

    def get_t_v_all_layers(self, batch):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        img_feat = batch['img_feat'] #[b,2,l,d]
        userid = batch['userid']
        img_mask = batch['img_mask']
        txt_mask = batch['txt_mask']

        
        _input, all_encoder_layers = self.uniter.get_all_layers(input_ids, position_ids,
                                      img_feat, img_mask, txt_mask, userid, 
                                      img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=True, layer_num=self.layer_num)

        t_fea_list = []
        v_fea_list = []
        all_encoder_layers.insert(0, _input)
        #index of texts：1+prefix_num: sum(txt_mask)+prefix_num
        #index of vision：sum(txt_mask)+prefix_num: sum(attn_mask)+prefix_num/ sum(txt_mask)+prefix_num+ sum(img_mask)
        bs = img_mask.shape[0]
        hidden_size = self.config.hidden_size
        img_nbb = torch.sum(img_mask,dim=-1)
        txt_num = torch.sum(txt_mask, dim=-1)

        for layer in all_encoder_layers:
            t_fea = torch.zeros((bs, hidden_size))
            v_fea = torch.zeros((bs, hidden_size))
            
            for i in range(bs):
                if self.config.add_prompt:
                    # print("num",layer[i].shape[1])
                    t_fea[i] = layer[i, 1+self.config.prefix_num: txt_num[i]+self.config.prefix_num].mean(dim=0)
                    v_fea[i] = layer[i, txt_num[i]+self.config.prefix_num: txt_num[i]+img_nbb[i]+self.config.prefix_num].mean(dim=0)
                else:
                    # print("num",layer[i].shape[1])
                    t_fea[i] = layer[i, 1: txt_num[i]].mean(dim=0)
                    v_fea[i] = layer[i, txt_num[i]: txt_num[i]+img_nbb[i]].mean(dim=0)
   
            t_fea = F.normalize(t_fea, p=2, dim=1)
            v_fea = F.normalize(v_fea, p=2, dim=1)
            t_fea_list.append(t_fea)
            v_fea_list.append(v_fea)
        
        return t_fea_list, v_fea_list

    def feature_pooling(self, inputs):
        if self.config.add_prompt:
            return inputs[:, 2:self.config.txt_len].mean(dim=1), inputs[:,self.config.txt_len:].mean(dim=1) 
        else:
            return inputs[:, 1:self.config.txt_len-1].mean(dim=1), inputs[:,self.config.txt_len:].mean(dim=1) 
       

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    #use pretrained model's first encoder to initialize prompt's encoder
    def load_encoder_for_prompt(self):
        new_dict = {}
        for k,v in self.uniter.state_dict().items():
            if k.startswith("encoder.layer.0."):
                name = k[16:]
                new_dict[name] = v
        missing_keys, unexpected_keys = self.uniter.prompt_module.encoder.load_state_dict(new_dict, strict=False)

        print("prompt missing_keys:\n{}".format(missing_keys))
        print("prompt unexpected_keys:\n{}".format(unexpected_keys))

