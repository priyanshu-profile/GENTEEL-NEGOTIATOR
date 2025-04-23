from lib2to3.pgen2.tokenize import tokenize
import os
import json
import torch
import pickle
from copy import deepcopy
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class RewardsDataset(data.Dataset):
    def __init__(self, args, data, tokenizer):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data["target"])
    
    def __getitem__(self, index):
        item = {}

        # context
        context_ids, context_token_type_ids, context_attention_mask = self.process_context(
            self.data["context"][index],
            self.data["context_strategy_seqs"][index],
            self.data["context_role"][index],
            self.data["context_positive_kts"][index],
            self.data["context_negative_kts"][index],
        )

        # target
        strategy = self.data["strategy"][index]
        target = "["+strategy+"]" + " " + " ".join(self.data["target"][index])
        target_kts = self.data["target_positive_kts"][index] + self.data["target_negative_kts"][index]
        target = target + " [KTS] " + " ".join(target_kts)
        target_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(target)) + [self.tokenizer.sep_token_id]
        target_token_type_ids = [1] * len(target_ids)
        target_attention_mask = [1] * len(target_ids)

        next_uttr = " ".join(self.data["next_uttr"][index])
        next_uttr_kts = self.data["next_uttr_positive_kts"][index] + self.data["next_uttr_negative_kts"][index]
        next_uttr = next_uttr + " [KTS] " + " ".join(next_uttr_kts)
        next_uttr_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(next_uttr)) + [self.tokenizer.sep_token_id]
        next_uttr_token_type_ids = [0] * len(next_uttr_ids)
        next_uttr_attention_mask = [1] * len(next_uttr_ids)

        if self.args.direction == "forward":
            item["input_ids"] = context_ids + target_ids
            item["token_type_ids"] = context_token_type_ids + target_token_type_ids
            item["attention_mask"] = context_attention_mask + target_attention_mask
        elif self.args.direction == "backward":
            item["input_ids"] = next_uttr_ids + target_ids
            item["token_type_ids"] = next_uttr_token_type_ids + target_token_type_ids
            item["attention_mask"] = next_uttr_attention_mask + target_attention_mask
        
        assert len(item["input_ids"]) == len(item["token_type_ids"]) == len(item["attention_mask"])

        item["label"] = self.data["label"][index]

        return item
    
    def process_context(self, context, context_strategy, context_role, context_pos_kts, context_neg_kts):
        context_ids = [self.tokenizer.cls_token_id]
        for uttr, strategy, role, pos_kts, neg_kts in zip(context, context_strategy, context_role, context_pos_kts, context_neg_kts):
            if role == "user":
                utterance = " ".join(uttr)
            elif role == "agent":
                utterance = "["+strategy+"]" + " " + " ".join(uttr)
            utterance += " [KTS] "
            kts = pos_kts + neg_kts
            utterance  = utterance + " ".join(kts)
            context_ids = context_ids + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)) + [self.tokenizer.sep_token_id]
        
        while len(context_ids) > self.args.max_context_length:
            cut_idx = context_ids.index(self.tokenizer.sep_token_id, -self.args.max_context_length+1)
            context_ids = [self.tokenizer.cls_token_id] + context_ids[cut_idx:]
        
        context_token_type_ids = [0] * len(context_ids)
        context_attenttion_mask = [1] * len(context_ids)
        
        return context_ids, context_token_type_ids, context_attenttion_mask
    
    def collate_fn(self, data):
        data_tensor = {}
        ignore_keys = {"target_txt", "context_txt"}
        for key in data[0].keys():
            if key in ignore_keys:
                data_tensor[key] = [item[key] for item in data]
            elif key == "label":
                data_tensor[key] = torch.tensor([item[key] for item in data], dtype=torch.long).to(self.args.device)
            elif key in ["target_lm_labels", "input_labels"]:
                data_tensor[key] = pad_sequence(
                    [torch.tensor(item[key], dtype=torch.long)
                    for item in data],
                    batch_first=True, padding_value=-100).to(self.args.device)
            elif key in ["context", "input"]:
                data_tensor[key] = pad_sequence(
                    [torch.tensor(item[key], dtype=torch.long)
                    for item in data],
                    batch_first=True, padding_value=self.PAD_id).to(self.args.device)
            else:
                data_tensor[key] = pad_sequence(
                    [torch.tensor(item[key], dtype=torch.long)
                    for item in data],
                    batch_first=True, padding_value=0).to(self.args.device)
        return data_tensor

