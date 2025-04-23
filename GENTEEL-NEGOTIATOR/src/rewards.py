import os
import json
import nltk
import torch
import pickle
import politenessr
import numpy as np
from .utils import kt_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
stopwords = stopwords.words("english")
from classifiers import contextual_negopredict, future_negopredict

class RewardAgent:
    def __init__(self, args, tokenizer, rewards_model):
        self.args = args
        self.total_kts = pickle.load(open(args.kts_vocab, "rb"))
        self.forward_rewards_model = rewards_model[0]
        self.backward_rewards_model = rewards_model[1]
        self.tokenizer = rewards_model[2]
        

    def process_context(self, context, context_nego_strategy, context_pol_strategy, context_role, context_pos_kts, context_neg_kts):
        context_ids = [self.tokenizer.cls_token_id]
        for uttr, nego_strategy, pol_strategy, role, pos_kts, neg_kts in zip(context, context_nego_strategy, context_pol_strategy, context_role, context_pos_kts, context_neg_kts):
            if role == "user":
                utterance = " ".join(uttr)
            elif role == "agent":
                utterance = "["+nego_strategy+"]" + " " + "["+pol_strategy+"]" + " " + " ".join(uttr)
            utterance += " [kts] "
            kts = pos_kts + neg_kts
            utterance  = utterance + " ".join(kts)
            context_ids = context_ids + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)) + [self.tokenizer.sep_token_id]
        
        while len(context_ids) > self.args.max_context_length:
            cut_idx = context_ids.index(self.tokenizer.sep_token_id, -self.args.max_context_length+1)
            context_ids = [self.tokenizer.cls_token_id] + context_ids[cut_idx:]
        
        context_token_type_ids = [0] * len(context_ids)
        context_attenttion_mask = [1] * len(context_ids)
        
        return context_ids, context_token_type_ids, context_attenttion_mask
    
    def process_response(self, uttr, kts, is_next_uttr=False):
        if is_next_uttr:
            utterance = " ".join(uttr) + " [kts] " + " ".join(kts)
            utterance_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)) + [self.tokenizer.sep_token_id]
            uttr_token_type_ids = [0] * len(utterance_ids)
        else:
            utterance = uttr + " [kts] " + " ".join(kts)  
            utterance_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)) + [self.tokenizer.sep_token_id]
            uttr_token_type_ids = [1] * len(utterance_ids)
        uttr_attention_mask = [1] * len(utterance_ids)
        return utterance_ids, uttr_token_type_ids, uttr_attention_mask
    
    @torch.no_grad()
    def calc_coherence(self, data, responses, response_kts):
        def rewards_model_dataset(idx, data, response, response_kts, is_forward=False, is_backward=False):
            response_ids, response_token_type_ids, response_attention_mask = self.process_response(
                response, 
                response_kts
            )
            if is_forward:
                context_ids, context_token_type_ids, context_attention_mask = self.process_context(
                    data["context_txt"][idx],
                    data["context_pol_strategy_seqs_txt"][idx],
                    data["context_role_txt"][idx],
                    data["context_positive_kts_txt"][idx],
                    data["context_negative_kts_txt"][idx]
                )
                input_ids = context_ids + response_ids
                token_type_ids = context_token_type_ids + response_token_type_ids
                attention_mask = context_attention_mask + response_attention_mask
            if is_backward:
                next_uttr_ids, next_uttr_token_type_ids, next_uttr_attention_mask = self.process_response(
                    data["next_uttr_txt"][idx],
                    data["next_uttr_positive_kts_txt"][idx] + data["next_uttr_negative_kts_txt"][idx],
                    is_next_uttr = True
                )
                input_ids = next_uttr_ids + response_ids
                token_type_ids = next_uttr_token_type_ids + response_token_type_ids
                attention_mask = next_uttr_attention_mask + response_attention_mask
            return torch.tensor(input_ids, dtype=torch.long).to(self.args.device), \
                   torch.tensor(token_type_ids, dtype=torch.long).to(self.args.device), \
                   torch.tensor(attention_mask, dtype=torch.long).to(self.args.device)
       
        context_last_infer_kts = data["context_last_infer_kts"]
        context_coherence_rewards = list()
        next_uttr_infer_kts = data["next_uttr_infer_kts"]
        future_coherence_rewards = list()
        for con_inf_kts, next_inf_kts, res_kts, (idx, response) in zip(context_last_infer_kts, next_uttr_infer_kts, response_kts, enumerate(responses)):
            # context
            con_occur_kts = [kts for kts in res_kts if kts in con_inf_kts]
            con_coher_reward = np.exp(len(con_occur_kts) / (len(res_kts) + 1e-20)) / np.exp(1)
            con_input_ids, con_token_type_ids, con_attention_mask = rewards_model_dataset(idx, data, response, res_kts, is_forward=True)
            con_outputs = self.forward_rewards_model(
                input_ids = con_input_ids.view(1, -1),
                token_type_ids = con_token_type_ids.view(1, -1),
                attention_mask = con_attention_mask.view(1, -1),
            )
            con_logits = torch.softmax(con_outputs["logits"], dim=1).cpu().numpy()
            con_coher_reward = con_coher_reward * con_logits[0][1]
            context_coherence_rewards.append(con_coher_reward)
            # future
            next_occur_kts = [kts for kts in res_kts if kts in next_inf_kts]
            next_coher_reward = np.exp(len(next_occur_kts) / (len(res_kts) + 1e-20)) / np.exp(1)
            next_input_ids, next_token_type_ids, next_attention_mask = rewards_model_dataset(idx, data, response, res_kts, is_backward=True)
            next_outputs = self.backward_rewards_model(
                input_ids = next_input_ids.view(1, -1),
                token_type_ids = next_token_type_ids.view(1, -1),
                attention_mask = next_attention_mask.view(1, -1)
            )
            next_logits = torch.softmax(next_outputs["logits"], dim=1).cpu().numpy()
            next_coher_reward = next_coher_reward * next_logits[0][1]
            future_coherence_rewards.append(next_coher_reward)
        return np.mean(context_coherence_rewards), np.array(context_coherence_rewards), \
               np.mean(future_coherence_rewards), np.array(future_coherence_rewards)

    def utterance_level_politeness(self, data, response_politeness_score):
        next_uttr_politeness_score = data["next_uttr_politeness_score"]
        dialog_turn = data["dialog_turn"]
        assert len(next_uttr_politeness_score) == len(response_politeness_score)
        
        utt_level_rewards = list()
        for utt, res_politeness_score, next_uttr_politeness_score in zip(dialog_turn, response_politeness_score, next_uttr_politeness_score):
            if utt >= self.args.max_dialog_turn:
                current_utt = self.args.max_dialog_turn - 1 + (turn - self.args.max_dialog_turn) / utt
            else:
                current_utt = utt
            reward = np.cos((np.pi*current_utt)/(2*self.args.max_dialog_turn)) * np.cos(np.pi * np.abs(res_politeness_score - next_uttr_politeness_score)/2)
            utt_level_rewards.append(reward)
        return np.mean(utt_level_rewards), np.array(utt_level_rewards)

    
    def dialogue_level_politeness(self, data, response_politeness_score):
        dialog_turn = data["dialog_turn"]
        context_user_sum_politeness_score = data["context_user_sum_politeness_score"]
        context_politeness_scores = data["context_politeness_scores"]
        context_roles = data["context_role"]
        assert len(dialog_turn) == len(context_user_sum_politeness_score) == len(response_politeness_score)
        dialogue_level_rewards = list()
        for current_max_turn, context_sum_politeness_socre, repsonse_pol_score, con_politeness_socres, con_roles in zip(dialog_turn, context_user_sum_politeness_score, response_politeness_score, context_politeness_scores, context_roles):
            reward = 0.0 # for calculating total reward
            #print('repsonse_pol_score',repsonse_pol_score)
            user_num = np.sum([1 for role in con_roles if role=="user"])
            utt = current_max_turn - user_num # for checking dialog turn
            assert len(con_politeness_socres) == len(con_roles)
            assert utt >= 0
            for politeness_score, role in zip(con_politeness_socres, con_roles):
                if role == "user":
                    turn += 1
                    assert utt <= current_max_turn
                    if utt >= self.args.max_dialog_turn:
                        current_utt = self.args.max_dialog_turn - 1 + (turn - self.args.max_dialog_turn) / utt
                    else:
                        current_utt = utt
                    reward += np.cos((np.pi*current_turn)/(2*self.args.max_dialog_turn)) * (repsonse_pol_score - politeness_score)
            dialogue_level_rewards.append(reward)
        return np.mean(dialogue_level_rewards), np.array(dialogue_level_rewards)



    def contextual_negotiation(self, data, response_negotiation_score):
        uttr_negotiation_score = data["uttr_negotiation_score"]
        dialog_turn = data["dialog_turn"]
        assert len(next_uttr_negotiation_score) == len(response_negotiation_score)
        
        utt_level_rewards = list()
        for utt, res_negotiation_score, uttr_negotiation_score in zip(dialog_turn, response_negotiation_score, uttr_negotiation_score):
            reward = uttr_negotiation_score - gamma_c * res_negotiation_score
            utt_level_rewards.append(reward)
    
        return np.mean(utt_level_rewards), np.array(utt_level_rewards)


    def future_negotiation(self, data, response_negotiation_score):
        next_uttr_negotiation_score = data["next_uttr_negotiation_score"]
        dialog_turn = data["dialog_turn"]
        assert len(next_uttr_negotiation_score) == len(response_negotiation_score)
        
        utt_level_rewards = list()
        for utt, res_negotiation_score, next_uttr_negotiation_score in zip(dialog_turn, response_negotiation_score, next_uttr_negotiation_score):
            reward = next_uttr_negotiation_score - gamma_f * res_negotiation_score
            utt_level_rewards.append(reward)
    
        return np.mean(utt_level_rewards), np.array(utt_level_rewards)


    def get_negotiation_score(self, utterance, flag):
        if flag == 1:
            negotiation_score = contextual_negopredict([utterance])
        else:
            negotiation_score = future_negopredict([utterance])
        return negotiation_score.item()

    def get_politeness_score(self, utterance):
        politeness_score = politenessr.predict([utterance])
        return politeness_score.item()

    def get_rewards(self, data, responses, is_evaluate=False):
        #negotiation rewards
        context_negotiation_score = [
            self.get_negotiation_score(utterance,1)
            for utterance in responses
        ]

        future_negotiation_score = [
            self.get_negotiation_score(utterance,0)
            for utterance in responses
        ]
        context_negotiation_reward, batch_context_nego_rewards = self.contextual_negotiation(data, context_negotiation_score)
        future_negotiation_reward, batch_future_nego_rewards = self.future_negotiation(data, future_negotiation_score)


        #politeness rewards
        response_utt_politeness_score = [
            self.get_politeness_score(utterance)
            for utterance in responses
        ]
        response_dial_politeness_score = [
            self.get_politeness_score(utterance)
            for utterance in responses
        ]
        utt_level_politeness_reward, batch_utt_level_rewards = self.utterance_level_politeness(data, response_utt_politeness_score)
        dial_level_politeness_reward, batch_dial_level_rewards = self.dialogue_level_politeness(data, response_dial_politeness_score)

        # word coherence rewards
        response_kts = [
            list(set([kts for kts in kt_tokenize(utterance) if kts in self.total_kts]))
            for utterance in responses
        ]
        
        context_coherence_reward, batch_context_rewards, \
        future_coherence_reward, batch_future_rewards = self.calc_coherence(data, responses, response_kts)

        eval_rewards = context_negotiation_reward + future_negotiation_reward + utt_level_politeness_reward + dial_level_politeness_reward + \
                context_coherence_reward + future_coherence_reward

        rewards = self.args.context_nego_reward_weight * context_negotiation_reward + \
                  self.args.future_nego_reward_weight * future_negotiation_reward + \
                  self.args.turn_reward_weight * utt_level_politeness_reward + \
                  self.args.conversation_reward_weight * dial_level_politeness_reward + \
                  self.args.context_reward_weight * context_coherence_reward + \
                  self.args.future_reward_weight * future_coherence_reward
        batch_rewards = self.args.context_nego_reward_weight * batch_context_nego_rewards + \
                        self.args.future_nego_reward_weight * batch_future_nego_rewards + \
                        self.args.turn_reward_weight * batch_utt_level_rewards + \
                        self.args.conversation_reward_weight * batch_dial_level_rewards + \
                        self.args.context_reward_weight * batch_context_rewards + \
                        self.args.future_reward_weight * batch_future_rewards
        # assert rewards == np.mean(batch_rewards)
        return rewards, batch_rewards, eval_rewards
    