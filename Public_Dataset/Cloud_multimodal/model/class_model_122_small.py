"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict

import torch
from torch import nn
from .model import UniterPreTrainedModel, UniterModel


class UniterForClassSmall(UniterPreTrainedModel):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, layer_num):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.classifier = nn.Linear(config.hidden_size, 122)
        self.apply(self.init_weights)
        self.layer_num = layer_num

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=True)
        sequence_output = sequence_output[self.layer_num-1]            
        pooled_output = self.uniter.pooler(sequence_output)
        outputs = self.classifier(pooled_output)
        return outputs



