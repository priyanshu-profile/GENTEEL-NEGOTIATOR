import os
import nltk
import json
import torch
import pickle
import numpy as np
from collections import defaultdict
from copy import deepcopy
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class Dataset(data.Dataset):
    def __init__(self, args, data, tokenizer, politeness_statistic, nego_strategy_statistic):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.CLS_id = tokenizer.cls_token_id
        self.EOS_id = tokenizer.eos_token_id
        self.SEP_id = tokenizer.sep_token_id
        self.politeness_statistic = politeness_statistic
        self.nego_strategy_statistic = nego_strategy_statistic

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}

        # context
        item["context"], item["context_role"], item["context_token_type"], \
        item["ori_context"], item["ori_context_role"], item["ori_context_token_type"] = self.process_context(
            self.data["context"][index],
            self.data["context_strategy_seqs"][index],
            self.data["context_role"][index],
            self.data["context_positive_kts"][index], 
            self.data["context_negative_kts"][index],
        )
        item["context_txt"] = self.data["context"][index]

        # infered kts
        item["context_infer_pos_kts"], item["context_infer_pos_role"], item["context_infer_pos_token_type"], \
        item["context_infer_neg_kts"], item["context_infer_neg_role"], item["context_infer_neg_token_type"], \
        item["next_uttr_infer_pos_kts"], item["next_uttr_infer_pos_role"], item["next_uttr_infer_pos_token_type"], \
        item["next_uttr_infer_neg_kts"], item["next_uttr_infer_neg_role"], item["next_uttr_inder_neg_token_typr"] = self.process_infered_kts(
            len(self.data["context"][index])+1,
            self.data["context_infer_pos_kts"][index], 
            self.data["context_infer_neg_kts"][index],
            self.data["next_uttr_infer_pos_kts"][index], 
            self.data["next_uttr_infer_neg_kts"][index]
        )

        # negotiation strategy labels
        item["context_nego_strategy_labels"], item["next_uttr_nego_strategy_labels"] = self.process_nego_strategy_labels(
            self.data["context_nego_strategy_labels"][index],
            self.data["next_uttr_nego_strategy_labels"][index]
        )

        # politeness labels
        item["context_pos_politeness_labels"], item["context_neg_politeness_labels"], item["next_uttr_pos_politeness_labels"], item["next_uttr_neg_politeness_labels"] = self.process_politeness_labels(
            self.data["context_pos_pol_labels"][index],
            self.data["context_neg_pol_labels"][index],
            self.data["next_uttr_pos_pol_labels"][index],
            self.data["next_uttr_neg_pol_labels"][index]
        )

        # kts labels
        item["target_pos_kts_labels"], item["target_neg_kts_labels"], item["next_uttr_pos_kts_labels"], item["next_uttr_neg_kts_labels"] = self.process_kts_labels(
            self.data["target_positive_kts"][index],
            self.data["target_negative_kts"][index],
            self.data["next_uttr_positive_kts"][index],
            self.data["next_uttr_negative_kts"][index]
        )
        
        # target
        item["target"], item["target_role"], item["target_lm_labels"] = self.process_target(
            self.data["target"][index], self.data["strategy"][index])
        # item["target_txt"] = ["[" + self.data["strategy"][index] +"]"] + self.data["target"][index]
        item["target_txt"] = self.data["target"][index]

        # rewards ready
        item["dialog_turn"] = self.data["dialog_turn"][index]
        item["context_nego_strategy_score"] = self.data["context_nego_strategy_score"][index]
        item["future_nego_strategy_score"] = self.data["future_nego_strategy_score"][index]
        item["context_user_sum_politeness_score"] = self.data["context_user_sum_politeness_score"][index]
        item["next_uttr_politeness_score"] = self.data["next_uttr_politeness_score"][index]
        item["context_last_infer_kts"] = self.data["context_last_infer_kts"][index]
        item["next_uttr_infer_kts"] = self.data["next_uttr_infer_kts"][index]

        item["context_txt"] = self.data["context"][index]
        item["context_nego_strategy_txt"] = self.data["context_nego_strategy"][index]
        item["future_nego_strategy_txt"] = self.data["future_nego_strategy"][index]
        item["context_strategy_seqs_txt"] = self.data["context_strategy_seqs"][index]
        item["context_role_txt"] = self.data["context_role"][index]
        item["context_positive_kts_txt"] = self.data["context_positive_kts"][index] 
        item["context_negative_kts_txt"] = self.data["context_negative_kts"][index]
        item["next_uttr_txt"] = self.data["next_uttr"][index]
        item["next_uttr_positive_kts_txt"] = self.data["next_uttr_positive_kts"][index]
        item["next_uttr_negative_kts_txt"] = self.data["next_uttr_negative_kts"][index]

        item["context_politeness_scores"] = self.data["context_politeness_scores"][index]
        item["context_role"] = self.data["context_role"][index]

        return item

    def process_context(self, context, context_strategy, context_role, context_positive_kts, context_negative_kts):
        context_ids = [self.CLS_id]
        context_role_ids = [self.CLS_id]
        context_token_type_ids = [self.CLS_id]
        # context uttr
        for (i, uttr), strategy, role in zip(enumerate(context), context_strategy, context_role):
            if role == "user":
                uttr_encode_ids = self.tokenizer.encode(" ".join(uttr)) + [self.EOS_id]
            elif role == "agent":
                uttr_encode_ids = self.tokenizer.encode("["+strategy+"]") + self.tokenizer.encode(" ".join(uttr)) + [self.EOS_id]
            else:
                raise ValueError("The Label of Role is Error!")

            context_ids += uttr_encode_ids
            spk = self.args.user_idx if role == "user" else self.args.agent_idx
            context_role_ids = context_role_ids + [spk] * len(uttr_encode_ids)
            context_token_type_ids = context_token_type_ids + [i+1] * len(uttr_encode_ids)

            if i == len(context)-1:
                context_ids += [self.SEP_id]
                context_role_ids += [spk]
                context_token_type_ids += [i+1]
        
        while len(context_ids) > self.args.max_context_length:
            cut_idx = context_ids.index(self.EOS_id, -self.args.max_context_length+1)
            context_ids = [self.CLS_id] + context_ids[cut_idx:]
            context_role_ids = [self.CLS_id] + context_role_ids[cut_idx:]
            context_token_type_ids = [self.CLS_id] + context_token_type_ids[cut_idx:]
        assert len(context_ids) == len(context_role_ids) == len(context_token_type_ids)
        
        ori_context_ids = deepcopy(context_ids)
        ori_context_role_ids = deepcopy(context_role_ids)
        ori_context_token_type_ids = deepcopy(context_token_type_ids)

        # context kts
        assert len(context_positive_kts) == len(context_negative_kts)
        context_kts_ids = []
        context_kts_role_ids = []
        context_kts_token_type_ids = []
        for (i, role), context_pos_kts, context_neg_kts in zip(enumerate(context_role), context_positive_kts, context_negative_kts):
            context_pos_neg_kts = list(set(context_pos_kts + context_neg_kts))
            pos_neg_kts_ids = self.tokenizer.encode(" ".join(context_pos_neg_kts)) + [self.EOS_id]
            context_kts_ids += pos_neg_kts_ids
            spk = self.args.context_kts_user_idx if role == "user" else self.args.context_kts_agent_idx
            context_kts_role_ids = context_kts_role_ids + [spk] * len(pos_neg_kts_ids)
            context_kts_token_type_ids = context_kts_token_type_ids + [i+1] * len(pos_neg_kts_ids)
        while len(context_kts_ids) > self.args.max_context_kts_length:
            cut_idx = context_kts_ids.index(self.EOS_id, -self.args.max_context_kts_length+1)
            context_kts_ids = context_kts_ids[cut_idx:]
            context_kts_role_ids = context_kts_role_ids[cut_idx:]
            context_kts_token_type_ids = context_kts_token_type_ids[cut_idx:]
        assert len(context_kts_ids) == len(context_kts_role_ids) == len(context_kts_token_type_ids)

        context_ids += context_kts_ids
        context_role_ids += context_kts_role_ids
        context_token_type_ids += context_kts_token_type_ids
        
        return context_ids, context_role_ids, context_token_type_ids, ori_context_ids, ori_context_role_ids, ori_context_token_type_ids

    def process_infered_kts(self, turn_id, context_infer_pos_kts, context_infer_neg_kts, next_uttr_infer_pos_kts, next_uttr_infer_neg_kts):
        def process_kts(infer_kts, role_id, turn_id):
            infer_kts = list(set([kts for _,_,kts,_ in infer_kts[:self.args.max_infer_kts_length]]))
            infer_kts_ids = self.tokenizer.encode(" ".join(infer_kts))
            infer_role_ids = len(infer_kts_ids) * [role_id]
            infer_token_type_ids = len(infer_kts_ids) * [turn_id]
            return infer_kts_ids, infer_role_ids, infer_token_type_ids
        context_infer_pos_kts_ids, context_infer_pos_role_ids, context_infer_pos_token_type_ids = process_kts(
            context_infer_pos_kts, self.args.context_infer_kts_idx, turn_id
        )
        context_infer_neg_kts_ids, context_infer_neg_role_ids, context_infer_neg_token_type_ids = process_kts(
            context_infer_neg_kts, self.args.context_infer_kts_idx, turn_id
        )
        next_uttr_infer_pos_kts_ids, next_uttr_infer_pos_role_ids, next_uttr_infer_pos_token_type_ids = process_kts(
            next_uttr_infer_pos_kts, self.args.next_uttr_infer_kts_idx, turn_id
        )
        next_uttr_infer_neg_kts_ids, next_uttr_infer_neg_role_ids, next_uttr_infer_neg_token_type_ids = process_kts(
            next_uttr_infer_neg_kts, self.args.next_uttr_infer_kts_idx, turn_id
        )
        return context_infer_pos_kts_ids, context_infer_pos_role_ids, context_infer_pos_token_type_ids, \
               context_infer_neg_kts_ids, context_infer_neg_role_ids, context_infer_neg_token_type_ids, \
               next_uttr_infer_pos_kts_ids, next_uttr_infer_pos_role_ids, next_uttr_infer_pos_token_type_ids, \
               next_uttr_infer_neg_kts_ids, next_uttr_infer_neg_role_ids, next_uttr_infer_neg_token_type_ids
    
    def process_nego_strategy_labels(self, context_nego_strategy, next_uttr_nego_strategy):
        context_nego_strategy_labels = [self.nego_strategy_statistic[nego_strategy]["idx"] for nego_strategy in context_nego_strategy]
        next_uttr_nego_strategy_labels = [self.nego_strategy_statistic[nego_strategy]["idx"] for nego_strategy in next_uttr_nego_strategy]
        return context_nego_strategy_labels, next_uttr_nego_strategy_labels

    def process_politeness_labels(self, context_pos_politeness, context_neg_politeness, next_uttr_pos_politeness, next_uttr_neg_politeness):
        context_pos_politeness_labels = [self.politeness_statistic[politeness]["idx"] for politeness in context_pos_politeness]
        context_neg_politeness_labels = [self.politeness_statistic[politeness]["idx"] for politeness in context_neg_politeness]
        next_uttr_pos_politeness_labels = [self.politeness_statistic[politeness]["idx"] for politeness in next_uttr_pos_politeness]
        next_uttr_neg_politeness_labels = [self.politeness_statistic[politeness]["idx"] for politeness in next_uttr_neg_politeness]
        return context_pos_politeness_labels, context_neg_politeness_labels, next_uttr_pos_politeness_labels, next_uttr_neg_politeness_labels

    def process_kts_labels(self, target_positive_kts, target_negative_kts, next_uttr_positive_kts, next_uttr_negative_kts):
        target_pos_kts_labels = self.tokenizer.encode(" ".join(target_positive_kts))
        target_neg_kts_labels = self.tokenizer.encode(" ".join(target_negative_kts))
        next_uttr_pos_kts_labels = self.tokenizer.encode(" ".join(next_uttr_positive_kts))
        next_uttr_neg_kts_labels = self.tokenizer.encode(" ".join(next_uttr_negative_kts))
        return target_pos_kts_labels, target_neg_kts_labels, next_uttr_pos_kts_labels, next_uttr_neg_kts_labels
    
    def process_target(self, target, strategy):
        target_ids = self.tokenizer.encode("["+strategy+"]") + self.tokenizer.encode(" ".join(target)) + [self.EOS_id]
        target_role_ids = [self.args.agent_idx] * len(target_ids)
        return target_ids, target_role_ids, target_ids

    def collate_fn(self, data):        
        data_tensor = {}
        ignore_keys = {"context_txt", "target_txt", "dialog_turn", "context_nego_strategy_score", "future_nego_strategy_score", "context_user_sum_politeness_score", "next_uttr_politeness_score", "context_last_infer_kts", "next_uttr_infer_kts", "context_role", "context_politeness_scores"}
        for key in data[0].keys():
            if key in ignore_keys or "txt" in key:
                data_tensor[key] = [item[key] for item in data]
            elif key == "target_lm_labels":
                data_tensor[key] = pad_sequence(
                    [torch.tensor(item[key], dtype=torch.long)
                    for item in data],
                    batch_first=True, padding_value=-100).to(self.args.device)
            else:
                data_tensor[key] = pad_sequence(
                    [torch.tensor(item[key], dtype=torch.long)
                    for item in data],
                    batch_first=True, padding_value=0).to(self.args.device)
        return data_tensor
