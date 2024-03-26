"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
from copy import deepcopy
import json
import logging
from io import open

import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
import torch.nn.functional as F

from .layer import BertLayer, BertPooler


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
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
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


class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

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

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model


class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class UniterModelPrompt(nn.Module):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim):
        super().__init__()
        self.config = config
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_mask, txt_mask, userid,
                                    img_pos_feat,
                                    attention_mask,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):

        #get prompt
        if self.config.add_prompt:
            # img_feat [b,2,l,d]
            # img_feat[:,0] unified, img_feat[:,1] personalized
            base_img_feat = img_feat[:,0].clone() 
            fine_img_feat = img_feat[:,1].clone()
            prompt = self.prompt_module(userid, base_img_feat, fine_img_feat, img_pos_feat, img_mask)

        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)

        #prepend prompt to textual tokens (transform embeddings,mask,gather_index)
        bs = txt_emb.shape[0]
        tl = txt_emb.shape[1]
        hidden = txt_emb.shape[2]
        if self.config.add_prompt and not self.config.prompt_and_embed:
            #prepend after [CLS] and before texts
            #text_emb's shape [b, tl_max, hidden]
            prompt_txt_emb = torch.zeros(bs, tl+self.config.prefix_num, hidden).to(txt_emb.device)
            prompt_txt_emb[:,0] = txt_emb[:,0] #cls
            prompt_txt_emb[:, 1:self.config.prefix_num+1] = prompt 
            prompt_txt_emb[:, self.config.prefix_num+1:] = txt_emb[:, 1:]
            txt_emb = prompt_txt_emb

            #attn_mask(the most token number in the batch)
            prompt_attn_masks = torch.zeros(bs, attention_mask.shape[1]+self.config.prefix_num).to(attention_mask.device)
            prompt_attn_masks[:, :self.config.prefix_num] = 1 #mask add 1, number of prefix
            prompt_attn_masks[:, self.config.prefix_num:] = attention_mask
            attention_mask = prompt_attn_masks

            #gather_index(from the start of image, each add prefix
            prompt_gather_index = torch.arange(0, gather_index.shape[1]+self.config.prefix_num, dtype=torch.long,).unsqueeze(0).repeat(bs, 1).to(gather_index.device)
            img_start = torch.sum(txt_mask, dim=-1)
            for i in range(img_start.shape[0]):
                prompt_gather_index[i, self.config.prefix_num + int(img_start[i]):] =  gather_index[i, int(img_start[i]):] + self.config.prefix_num
            gather_index = prompt_gather_index

        if self.config.add_embed and not self.config.prompt_and_embed:
            #insert user_embed
            user_embed = self.embed_module(userid) #[b,p]
            #txt_emb
            prompt_txt_emb = torch.zeros(bs, tl+self.config.embed_num, hidden).to(txt_emb.device)
            prompt_txt_emb[:,0] = txt_emb[:,0] #cls
            prompt_txt_emb[:, 1:1+self.config.embed_num] = user_embed
            prompt_txt_emb[:, 1+self.config.embed_num:] = txt_emb[:, 1:]
            txt_emb = prompt_txt_emb

            #mask
            prompt_attn_masks = torch.zeros(bs, attention_mask.shape[1]+self.config.embed_num).to(attention_mask.device)
            prompt_attn_masks[:, :self.config.embed_num] = 1 #mask add 1, number of prefix
            prompt_attn_masks[:, self.config.embed_num:] = attention_mask
            attention_mask = prompt_attn_masks

            #gather_index
            prompt_gather_index = torch.arange(0, gather_index.shape[1]+self.config.embed_num, dtype=torch.long,).unsqueeze(0).repeat(bs, 1).to(gather_index.device)
            img_start = torch.sum(txt_mask, dim=-1)
            for i in range(img_start.shape[0]):
                prompt_gather_index[i, self.config.embed_num+int(img_start[i]):] = gather_index[i, int(img_start[i]):] + self.config.embed_num
            gather_index = prompt_gather_index

        if self.config.prompt_and_embed:
            user_embed = self.embed_module(userid)

            prompt_txt_emb = torch.zeros(bs, tl+self.config.prefix_num+self.config.embed_num, hidden).to(txt_emb.device)
            prompt_txt_emb[:,0] = txt_emb[:,0] #cls
            prompt_txt_emb[:, 1:self.config.embed_num+1] = user_embed
            prompt_txt_emb[:, 1+self.config.embed_num:self.config.embed_num+self.config.prefix_num+1] = prompt 
            prompt_txt_emb[:, self.config.embed_num+self.config.prefix_num+1:] = txt_emb[:, 1:]
            txt_emb = prompt_txt_emb

            #attn_mask
            prompt_attn_masks = torch.zeros(bs, attention_mask.shape[1]+self.config.embed_num+self.config.prefix_num).to(attention_mask.device)
            prompt_attn_masks[:, :self.config.embed_num+self.config.prefix_num] = 1 #mask add 1, number of prefix
            prompt_attn_masks[:, self.config.embed_num+self.config.prefix_num:] = attention_mask
            attention_mask = prompt_attn_masks

            #gather_index
            prompt_gather_index = torch.arange(0, gather_index.shape[1]+self.config.embed_num+self.config.prefix_num, dtype=torch.long,).unsqueeze(0).repeat(bs, 1).to(gather_index.device)
            img_start = torch.sum(txt_mask, dim=-1)
            for i in range(img_start.shape[0]):
                prompt_gather_index[i, self.config.embed_num+self.config.prefix_num + int(img_start[i]):] =  gather_index[i, int(img_start[i]):] + self.config.embed_num+self.config.prefix_num
            gather_index = prompt_gather_index

        # original unified visual features
        img_emb = self._compute_img_embeddings(
            img_feat[:,0], img_pos_feat, img_type_ids)

        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return attention_mask, gather_index, embedding_output

    #img_feat: [b,2,l,d] 
    def forward(self, input_ids, position_ids,
                img_feat, img_mask, txt_mask, userid,
                img_pos_feat, 
                attention_mask, gather_index=None, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):

        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            attention_mask, gather_index, embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_mask, txt_mask, userid, #new inputs
                img_pos_feat,
                attention_mask, #to change mask
                gather_index, img_masks, txt_type_ids, img_type_ids)

        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers

    def get_all_layers(self, input_ids, position_ids,
                            img_feat, img_mask, txt_mask, userid,
                            img_pos_feat, 
                            attention_mask, gather_index=None, img_masks=None,
                            output_all_encoded_layers=True,
                            txt_type_ids=None, img_type_ids=None, layer_num=3):
        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            attention_mask, gather_index, embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_mask, txt_mask, userid, #new inputs
                img_pos_feat,
                attention_mask, #to change mask
                gather_index, img_masks, txt_type_ids, img_type_ids)

        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        else:
            encoded_layers = encoded_layers[:layer_num]

        return embedding_output, encoded_layers

    def get_prompt(self, input_ids, position_ids,
                    img_feat, img_mask, txt_mask, userid,
                    img_pos_feat, 
                    attention_mask, gather_index=None, img_masks=None,
                    output_all_encoded_layers=True,
                    txt_type_ids=None, img_type_ids=None):

        base_img_feat = img_feat[:,0].clone() 
        fine_img_feat = img_feat[:,1].clone()
        prompt = self.prompt_module(userid, base_img_feat, fine_img_feat, img_pos_feat, img_mask)
        return prompt

    def construct_prompt(self):
        prompt_config = UniterConfig.from_json_file('/mnt/workspace/UNITER-master/config/uniter-prompt.json')
        if self.config.light_prompt:
            prompt_config.hidden_size = prompt_config.prompt_hidden_size
            prompt_config.intermediate_size = prompt_config.prompt_intermediate_size
            prompt_config.num_attention_heads = prompt_config.prompt_num_attention_heads

        if self.config.prompt_input == "two":
            self.prompt_module = Prompt_two(prompt_config)
        elif self.config.prompt_input == "one":
            self.prompt_module = Prompt_one(prompt_config)
        elif self.config.prompt_input == "diff":
            self.prompt_module = Prompt_diff(prompt_config)
        elif self.config.prompt_input == "base":
            self.prompt_module = Prompt_base(prompt_config)

        self.prompt_module.cuda()
    
    def construct_embed(self):
        self.embed_module = UserEmbed(self.config)
        self.embed_module.cuda()

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

class UserEmbed(nn.Module):
    """docstring for Prompt
    input：
    userid #[b,], mapped userid(0,1,..)
    output：
    user_embed
    (mapped to hidden_state_dim if needed)
    """
    def __init__(self, config):
        super(UserEmbed, self).__init__()
        self.config = config

        if self.config.embed_num == 1:
            self.embed_module = nn.Embedding(self.config.user_num, self.config.embed_dim)
            self.embed_linear = None
            self.embed_layer_norm = None
            if self.config.embed_dim != self.config.hidden_size:
                self.embed_linear = nn.Linear(self.config.embed_dim, self.config.hidden_size)
            if self.config.norm_embed:
                self.embed_layer_norm = FusedLayerNorm(self.config.hidden_size, eps=1e-12)
        else:
            self.embed_module = nn.ModuleList([nn.Embedding(self.config.user_num, self.config.embed_dim) for i in range(self.config.embed_num)])
            if self.config.embed_dim != self.config.hidden_size:
                self.embed_linear = nn.ModuleList([nn.Linear(self.config.embed_dim, self.config.hidden_size) for i in range(self.config.embed_num)])
            if self.config.norm_embed:
                self.embed_layer_norm = FusedLayerNorm(self.config.hidden_size, eps=1e-12)

        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  
    
    def forward(self, userid):
        if self.config.embed_num == 1:
            user_embed = self.embed_module(userid) #[b,d]
            if self.embed_linear is not None:
                user_embed = self.embed_linear(user_embed)
            if self.embed_layer_norm is not None:
                user_embed = self.embed_layer_norm(user_embed)
            user_embed = user_embed.unsqueeze(1) #[b,1,d]
        else:
            user_embed = torch.cat([(self.embed_module[i](userid)).unsqueeze(1) for i in range(self.config.embed_num)], dim=1) #[b,l,d]
            if self.config.embed_dim != self.config.hidden_size:
                user_embed = torch.cat([self.embed_linear[i](user_embed[:,i,:]).unsqueeze(1) for i in range(self.config.embed_num)], dim=1)
            if self.config.norm_embed:
                user_embed = self.embed_layer_norm(user_embed)
        return user_embed

class Prompt_two(nn.Module):
    """docstring for Prompt
    input：
    base_feat, fine_feat,  #[128, 224] #data['live_ts_embed']
    live_mask:[128]0 or 1 #data['live_embed_mask']
    pos_feat
    """
    def __init__(self, config):
        super(Prompt_two, self).__init__()
        self.config = config

        #prompt
        self.sep = nn.Parameter(torch.zeros(self.config.frame_feat_dim)) #embedding of [SEP]
        nn.init.normal_(self.sep, mean=0.0, std=self.config.initializer_range) #init

        self.sep_pos = nn.Parameter(torch.zeros(7)) #position of [SEP]
        nn.init.normal_(self.sep_pos, mean=0.0, std=self.config.initializer_range) #init

        if self.config.per_token: #refix_num
            if self.config.prefix_num == 1: 
                self.prompt_embedding = nn.Embedding(self.config.user_num, self.config.per_token_dim)
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
            elif self.config.prefix_num > 1:
                self.prompt_embedding = nn.ModuleList([nn.Embedding(self.config.user_num, self.config.per_token_dim) for i in range(self.config.prefix_num)])
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
        else:
            self.prompt_embedding = nn.Embedding(self.config.prefix_num, self.config.hidden_size)#[CLS]

        #embedding
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.type_embedding = nn.Embedding(2, self.config.hidden_size) #0:baseline, 1:finetune 
        self.pos_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.base_linear = nn.Linear(self.config.frame_feat_dim, self.config.hidden_size) #baseline
        self.base_img_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.base_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)

        self.fine_linear = nn.Linear(self.config.frame_feat_dim, self.config.hidden_size) #finetune
        self.fine_img_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.fine_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)

        #encoder
        self.encoder = BertLayer(self.config)

        if self.config.hidden_size != 768:
            self.last_linear = nn.Linear(self.config.hidden_size, 768)

        self.apply(self.init_weights)
        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    #add [SEP] to end of unified visual features,  fea:[b,l,d]; mask:[b,l]
    def insert_sep(self, fea, mask):
        base_mask = torch.zeros(mask.shape[0], mask.shape[1]+1).to(mask.device)
        base_fea = torch.zeros(fea.shape[0], fea.shape[1]+1, fea.shape[2]).to(mask.device)
        base_fea[:,:-1] = fea
        nbb = torch.sum(mask, dim=-1)
        # print(("nbb", nbb))
        for i in range(mask.shape[0]):
            base_mask[i, :int(nbb[i])+1] = 1 #1 of (nbb+1)
            base_fea[i, int(nbb[i])] = self.sep #(nbb+1)-th

        return base_fea, base_mask, mask

    #input for encoder
    def get_embed_input(self, userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask):
        trainsformed_sep_pos = self.pos_norm(self.pos_linear(self.sep_pos.repeat(userid.shape[0],1,1))) 
        transformed_pos = self.pos_norm(self.pos_linear(img_pos_feat)) 
        base_transformed_pos = torch.cat([transformed_pos, trainsformed_sep_pos], dim=1)

        #baseline
        base_type_ids = torch.zeros((base_img_feat.shape[0], base_img_feat.shape[1]+1),dtype=torch.int).cuda()#[b,l+1,1]
        base_type_embeddings = self.type_embedding(base_type_ids)

        base_fea, base_mask, fine_mask = self.insert_sep(base_img_feat, live_embed_mask) 
        base_transformed_im = self.base_img_norm(self.base_linear(base_fea))

        base_embeddings = base_transformed_im + base_transformed_pos + base_type_embeddings
        base_embeddings = self.base_layer_norm(base_embeddings)
        base_embeddings = self.dropout(base_embeddings)

        #fine
        fine_type_ids = torch.ones_like(fine_img_feat[:, :, 0], dtype=torch.int).cuda() #[b,l,1]填充1
        fine_type_embeddings = self.type_embedding(fine_type_ids)

        fine_transformed_im = self.fine_img_norm(self.fine_linear(fine_img_feat))
        fine_embeddings = fine_transformed_im + transformed_pos + fine_type_embeddings
        fine_embeddings = self.fine_layer_norm(fine_embeddings)
        fine_embeddings = self.dropout(fine_embeddings)

        #prompt
        batch_size = base_img_feat.shape[0]

        if self.config.per_token:
            if self.config.prefix_num == 1:
                prompt  = self.prompt_embedding(userid)
                if self.prompt_linear is not None:
                    prompt = self.prompt_linear(prompt)
                prompt = prompt.unsqueeze(1)
            elif self.config.prefix_num > 1:
                if self.prompt_linear is not None:
                    prompt = torch.cat([self.prompt_linear(self.prompt_embedding[i](userid)).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
                else:
                    prompt = torch.cat([self.prompt_embedding[i](userid).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
        else:
            prompt = self.prompt_embedding(torch.arange(self.config.prefix_num).to(base_img_feat.device)).repeat([batch_size,1,1]) #[b, prefix_num, d]
        
        #concat embed and attention_mask
        prompt_mask = torch.ones(batch_size, self.config.prefix_num).to(base_img_feat.device) #[b, p], p:prefix_num
        attention_mask = torch.cat([prompt_mask, base_mask, fine_mask], dim = 1) #[b, 2l+p]
        embed = torch.cat([prompt, base_embeddings, fine_embeddings], dim = 1) #[b, 2l+p, d]

        return embed, attention_mask

    #return prompt
    def forward(self, userid, base_img_feat, fine_img_feat, img_pos_feat, #vision
                live_embed_mask): #mask
        embed, attention_mask = self.get_embed_input(userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask)

        #compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 #

        encoder_output = self.encoder(embed, extended_attention_mask)
        prompt = encoder_output[:,:self.config.prefix_num,:]

        if self.config.hidden_size != 768:
            prompt = self.last_linear(prompt)

        return prompt

class Prompt_one(nn.Module):
    def __init__(self, config):
        super(Prompt_one, self).__init__()
        self.config = config
    
        if self.config.per_token: 
            if self.config.prefix_num == 1: 
                self.prompt_embedding = nn.Embedding(self.config.user_num, self.config.per_token_dim)
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
            elif self.config.prefix_num > 1:
                self.prompt_embedding = nn.ModuleList([nn.Embedding(self.config.user_num, self.config.per_token_dim) for i in range(self.config.prefix_num)])
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
        else:
            self.prompt_embedding = nn.Embedding(self.config.prefix_num, self.config.hidden_size)

        #embedding
        self.pos_linear = nn.Linear(7, config.hidden_size) 
        self.pos_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fine_linear = nn.Linear(self.config.frame_feat_dim, self.config.hidden_size) 
        self.fine_img_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.fine_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)

        #encoder
        self.encoder = BertLayer(self.config)

        self.apply(self.init_weights)
        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_embed_input(self, userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask):
        #transform personalized visual features（no type）
        transformed_pos = self.pos_norm(self.pos_linear(img_pos_feat))
        #fine
        fine_fea = fine_img_feat
        fine_mask = live_embed_mask
        fine_transformed_im = self.fine_img_norm(self.fine_linear(fine_fea))
        fine_embeddings = fine_transformed_im + transformed_pos 
        fine_embeddings = self.fine_layer_norm(fine_embeddings)
        fine_embeddings = self.dropout(fine_embeddings)

        batch_size = base_img_feat.shape[0]

        if self.config.per_token:
            if self.config.prefix_num == 1:
                prompt  = self.prompt_embedding(userid)
                if self.prompt_linear is not None:
                    prompt = self.prompt_linear(prompt)
                prompt = prompt.unsqueeze(1)
            elif self.config.prefix_num > 1:
                if self.prompt_linear is not None:
                    prompt = torch.cat([self.prompt_linear(self.prompt_embedding[i](userid)).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
                else:
                    prompt = torch.cat([self.prompt_embedding[i](userid).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
        else:
            prompt = self.prompt_embedding(torch.arange(self.config.prefix_num).to(base_img_feat.device)).repeat([batch_size,1,1]) #[b, prefix_num, d]

        prompt_mask = torch.ones(batch_size, self.config.prefix_num).to(base_img_feat.device) #[b, p], p;prefix_num
        attention_mask = torch.cat([prompt_mask, fine_mask], dim = 1) #[b, l+p]
        embed = torch.cat([prompt, fine_embeddings], dim = 1) #[b, l+p, d]

        return embed, attention_mask

    def forward(self, userid, base_img_feat, fine_img_feat, img_pos_feat, #vision
                live_embed_mask): #mask
        embed, attention_mask = self.get_embed_input(userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask)

        #compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 

        encoder_output = self.encoder(embed, extended_attention_mask)
        prompt = encoder_output[:,:self.config.prefix_num,:]

        return prompt

class Prompt_diff(nn.Module):
    def __init__(self, config):
        super(Prompt_diff, self).__init__()
        self.config = config

        if self.config.per_token: 
            if self.config.prefix_num == 1: 
                self.prompt_embedding = nn.Embedding(self.config.user_num, self.config.per_token_dim)
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
            elif self.config.prefix_num > 1:
                self.prompt_embedding = nn.ModuleList([nn.Embedding(self.config.user_num, self.config.per_token_dim) for i in range(self.config.prefix_num)])
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
        else:
            self.prompt_embedding = nn.Embedding(self.config.prefix_num, self.config.hidden_size)

        #embedding
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.pos_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fine_linear = nn.Linear(self.config.frame_feat_dim, self.config.hidden_size) 
        self.fine_img_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.fine_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)

        #encoder
        self.encoder = BertLayer(self.config)

        self.apply(self.init_weights)
        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_embed_input(self, userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask):
        transformed_pos = self.pos_norm(self.pos_linear(img_pos_feat)) 
        #diff
        diff_img_feat = fine_img_feat - base_img_feat #diff
        # fine_fea, fine_mask = self.insert_sep(diff_img_feat, live_embed_mask) 
        fine_fea = diff_img_feat
        fine_mask = live_embed_mask

        fine_transformed_im = self.fine_img_norm(self.fine_linear(fine_fea))
        fine_embeddings = fine_transformed_im + transformed_pos 
        fine_embeddings = self.fine_layer_norm(fine_embeddings)
        fine_embeddings = self.dropout(fine_embeddings)
        
        batch_size = base_img_feat.shape[0]

        if self.config.per_token:
            if self.config.prefix_num == 1:
                prompt  = self.prompt_embedding(userid)
                if self.prompt_linear is not None:
                    prompt = self.prompt_linear(prompt)
                prompt = prompt.unsqueeze(1)
            elif self.config.prefix_num > 1:
                if self.prompt_linear is not None:
                    prompt = torch.cat([self.prompt_linear(self.prompt_embedding[i](userid)).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
                else:
                    prompt = torch.cat([self.prompt_embedding[i](userid).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
        else:
            prompt = self.prompt_embedding(torch.arange(self.config.prefix_num).to(base_img_feat.device)).repeat([batch_size,1,1]) #[b, prefix_num, d]

        prompt_mask = torch.ones(batch_size, self.config.prefix_num).to(base_img_feat.device) #[b, p], p:prefix_num
        attention_mask = torch.cat([prompt_mask, fine_mask], dim = 1) #[b, l+p]
        embed = torch.cat([prompt, fine_embeddings], dim = 1) #[b, l+p, d]

        return embed, attention_mask

    def forward(self, userid, base_img_feat, fine_img_feat, img_pos_feat, 
                live_embed_mask): 
        embed, attention_mask = self.get_embed_input(userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask)

        #compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 

        encoder_output = self.encoder(embed, extended_attention_mask)
        prompt = encoder_output[:,:self.config.prefix_num,:]

        return prompt

class Prompt_base(nn.Module):
    def __init__(self, config):
        super(Prompt_base, self).__init__()
        self.config = config

        if self.config.per_token:
            if self.config.prefix_num == 1: 
                self.prompt_embedding = nn.Embedding(self.config.user_num, self.config.per_token_dim)
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
            elif self.config.prefix_num > 1:
                self.prompt_embedding = nn.ModuleList([nn.Embedding(self.config.user_num, self.config.per_token_dim) for i in range(self.config.prefix_num)])
                if self.config.per_token_dim != self.config.hidden_size:
                    self.prompt_linear = nn.Linear(self.config.per_token_dim, self.config.hidden_size)
                else:
                    self.prompt_linear = None
        else:
            self.prompt_embedding = nn.Embedding(self.config.prefix_num, self.config.hidden_size)

        #embedding
        self.pos_linear = nn.Linear(7, config.hidden_size) 
        self.pos_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.base_linear = nn.Linear(self.config.frame_feat_dim, self.config.hidden_size) 
        self.base_img_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.base_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)

        #encoder
        self.encoder = BertLayer(self.config)

        self.apply(self.init_weights)
        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_embed_input(self, userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask):
        transformed_pos = self.pos_norm(self.pos_linear(img_pos_feat)) 
        base_transformed_im = self.base_img_norm(self.base_linear(base_img_feat))
        base_embeddings = base_transformed_im + transformed_pos 
        base_embeddings = self.base_layer_norm(base_embeddings)
        base_embeddings = self.dropout(base_embeddings)

        batch_size = base_img_feat.shape[0]
        if self.config.per_token:
            if self.config.prefix_num == 1:
                prompt  = self.prompt_embedding(userid)
                if self.prompt_linear is not None:
                    prompt = self.prompt_linear(prompt)
                prompt = prompt.unsqueeze(1)
            elif self.config.prefix_num > 1:
                if self.prompt_linear is not None:
                    prompt = torch.cat([self.prompt_linear(self.prompt_embedding[i](userid)).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
                else:
                    prompt = torch.cat([self.prompt_embedding[i](userid).unsqueeze(1) for i in range(self.config.prefix_num)], dim = 1)
        else:
            prompt = self.prompt_embedding(torch.arange(self.config.prefix_num).to(base_img_feat.device)).repeat([batch_size,1,1]) #[b, prefix_num, d]

        prompt_mask = torch.ones(batch_size, self.config.prefix_num).to(base_img_feat.device) 
        attention_mask = torch.cat([prompt_mask, live_embed_mask], dim = 1) #[b, l+p]
        embed = torch.cat([prompt, base_embeddings], dim = 1) #[b, l+p, d]

        return embed, attention_mask

    def forward(self, userid, base_img_feat, fine_img_feat, img_pos_feat, 
                live_embed_mask): 
        embed, attention_mask = self.get_embed_input(userid, base_img_feat, fine_img_feat, img_pos_feat, live_embed_mask)

        #compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 

        encoder_output = self.encoder(embed, extended_attention_mask)
        prompt = encoder_output[:,:self.config.prefix_num,:]

        return prompt