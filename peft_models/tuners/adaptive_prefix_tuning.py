# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import enum
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Union
from collections import defaultdict

import torch
import torch.nn.functional as F

from transformers import BertForSequenceClassification, BertTokenizer, T5Tokenizer, BertConfig

from ..utils import PeftType, PromptLearningConfig



from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from ..utils import PeftType, PromptLearningConfig


### NOT YET COMPLETED 
@dataclass
class AdaptivePrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.ADAPTIVE_PREFIX_TUNING


# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
class PrefixRoutingEncoder(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)


    def forward(self, indices, input_ids, inputs_embeds, attention_mask, base_model=None):
        batch_size = inputs_embeds.shape[0]

        hidden_states = inputs_embeds
        sentence_sum = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
        non_zero_count = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1)
        sentence_embeds = sentence_sum / non_zero_count.float()

        

        
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
                

        return prompt_embeddings

    def save_load_information(self, data_idx, split=None):
        data_idx = data_idx.tolist()
        if split is None:
            split = "Train" if self.training else "Validation"
        for i, index in enumerate(data_idx):
            self.load_infos[split][index].append(self.load_routes[i].item())

    # --- code for token-level analysis --- #
    def activate_analysis(self):
        self.analysis = True

    def disable_analysis(self):
        self.analysis = False

    def fix_prompt(self, index):
        self.prompt_index = index

    def fix_token(self, token_index, prompt_index):
        self.token_index = token_index
        self.prompt_index = prompt_index
    
    def print_and_reset_load_counts(self):
        print(self.load_counts.long())
        self.load_counts.fill_(0)
        self.probs_sum.fill_(0)

    def reset_load_counts(self):
        self.load_counts.fill_(0)
        self.probs_sum.fill_(0)
    
