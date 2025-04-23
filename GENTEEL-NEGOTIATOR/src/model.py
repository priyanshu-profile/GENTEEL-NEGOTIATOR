from copy import deepcopy
import os
from re import I
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
from .utils import Attention
from .rewards import RewardAgent
from src.transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    #AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from src.transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig)

class GenteelNegotiator(nn.Module):
    def __init__(
        self, 
        args,
        tokenizer,
        config,
        nego_strategy_statistic,
        politeness_statistic,
        kts_vocab,
        rewards_model,
        ):
        super(Supporter, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.llama3_1 = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            config = 'facebook/meta-llama/Meta-Llama-3.1-8B-Instruct',
        )
        self.llama3_1.resize_token_embeddings(len(tokenizer))
        self.config = self.llama3_1.config
        # kts vocabulary
        self.kts_vocab = torch.tensor(tokenizer.encode(" ".join(kts_vocab)), dtype=torch.long).to(args.device)
        # mixture-of-experts
        self.negotiation_experts = nn.ModuleList([nn.Linear(in_features=args.emb_dim, out_features=args.emb_dim) for _ in range(args.num_negotiation_experts)])
        self.negotiation_strategy_classifier = nn.ModuleList([nn.Linear(in_features=args.emb_dim, out_features=len(nego_strategy_statistic)+1) for _ in range(args.num_negotiation_experts)])
        self.politeness_experts = nn.ModuleList([nn.Linear(in_features=args.emb_dim, out_features=args.emb_dim) for _ in range(args.num_politeness_experts)])
        self.politeness_classifier = nn.ModuleList([nn.Linear(in_features=args.emb_dim, out_features=len(politeness_statistic)+1) for _ in range(args.num_politeness_experts)])
        self.kts_experts = nn.ModuleList([nn.Linear(in_features=args.emb_dim, out_features=args.emb_dim) for _ in range(args.num_kts_experts)])
        self.kts_merge = nn.ModuleList([nn.Sequential(nn.Linear(in_features=2*args.emb_dim, out_features=args.emb_dim), nn.Tanh()) for _ in range(args.num_kts_experts)])
        self.kts_attention = nn.ModuleList([Attention(
                                query_size = args.emb_dim,
                                memory_size = args.emb_dim,
                                hidden_size = args.emb_dim,
                                mode = "mlp") 
                                for _ in range(args.num_kts_experts)
                             ])
        self.kts_prediction = nn.ModuleList([nn.Linear(in_features=args.emb_dim, out_features=self.config.vocab_size) for _ in range(args.num_kts_experts)])
        # Actor-Critic
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=args.emb_dim*(args.max_num_actions), out_features=args.emb_dim*int(args.max_num_actions/2)),
            nn.ELU(),
            nn.Dropout(p=args.policy_dropout),
            nn.Linear(in_features=args.emb_dim*int(args.max_num_actions/2), out_features=args.emb_dim),
            nn.ELU(),
            nn.Dropout(p=args.policy_dropout),
        )
        self.actor_network = nn.Linear(in_features=args.emb_dim, out_features=args.num_politeness_experts+args.num_kts_experts)
        self.critic_network = nn.Linear(in_features=args.emb_dim, out_features=1)
        self.expert_merge = nn.Linear(in_features=args.emb_dim*(args.num_politeness_experts+args.num_kts_experts), out_features=args.emb_dim)
        # reward agent
        self.reward_agent = RewardAgent(args, tokenizer, rewards_model)
        # activation function
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)
        # criterion
        self.criterion_ce = nn.CrossEntropyLoss(reduce=False)
        self.criterion_nll = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
        self.criterion_mse = nn.MSELoss()
    
    def negotiation_strategy_classification_task(
        self, 
        data, 
        context_negotiation_expert, 
        context_negotiation_expert, 
        next_uttr_negotiation_expert, 
        next_uttr_negotiation_expert
        ):
        context_nego_strategy_labels = data["context_nego_strategy_labels"]
        next_uttr_nego_strategy_labels = data["next_uttr_nego_strategy_labels"]
        negotiation_task_loss = 0.0
        negotiation_experts = [context_negotiation_expert, next_uttr_negotiation_expert]
        nego_strategy_labels = [context_nego_strategy_labels, next_uttr_nego_strategy_labels]
        for (idx, expert), labels in zip(enumerate(negotiation_experts), negotiation_labels):
            expert = expert.unsqueeze(1).repeat(1, labels.size(1), 1)
            expert_scores = self.negotiation_strategy_classifier[idx](expert)
            expert_loss = self.compute_multi_task_loss(expert_scores, labels)
            label_masks = torch.sign(labels).float()
            expert_loss = torch.sum(expert_loss * label_masks.flatten()) / (torch.sum(label_masks.flatten(0)) + 1e-20)
            negotiation_task_loss += expert_loss
        return negotiation_task_loss

    def politeness_classification_task(
        self, 
        data, 
        context_pos_politeness_expert, 
        context_neg_politeness_expert, 
        next_uttr_pos_politeness_expert, 
        next_uttr_neg_politeness_expert
        ):
        context_pos_politeness_labels = data["context_pos_politeness_labels"]
        context_neg_politeness_labels = data["context_neg_politeness_labels"]
        next_uttr_pos_politeness_labels = data["next_uttr_pos_politeness_labels"]
        next_uttr_neg_politeness_labels = data["next_uttr_neg_politeness_labels"]
        politeness_task_loss = 0.0
        politeness_experts = [context_pos_politeness_expert, context_neg_politeness_expert, next_uttr_pos_politeness_expert, next_uttr_neg_politeness_expert]
        politeness_labels = [context_pos_politeness_labels, context_neg_politeness_labels, next_uttr_pos_politeness_labels, next_uttr_neg_politeness_labels]
        for (idx, expert), labels in zip(enumerate(politeness_experts), politeness_labels):
            expert = expert.unsqueeze(1).repeat(1, labels.size(1), 1)
            expert_scores = self.politeness_classifier[idx](expert)
            expert_loss = self.compute_multi_task_loss(expert_scores, labels)
            label_masks = torch.sign(labels).float()
            expert_loss = torch.sum(expert_loss * label_masks.flatten()) / (torch.sum(label_masks.flatten(0)) + 1e-20)
            politeness_task_loss += expert_loss
        return politeness_task_loss

    def keyword_prediction_task(
        self, 
        data, 
        context_pos_kts_expert, 
        context_neg_kts_expert, 
        next_uttr_pos_kts_expert, 
        next_uttr_neg_kts_expert
        ):
        target_pos_kts_labels = data["target_pos_kts_labels"]
        target_neg_kts_labels = data["target_neg_kts_labels"]
        next_uttr_pos_kts_labels = data["next_uttr_pos_kts_labels"]
        next_uttr_neg_kts_labels = data["next_uttr_neg_kts_labels"]
        kts_labels = [target_pos_kts_labels, target_neg_kts_labels, next_uttr_pos_kts_labels, next_uttr_neg_kts_labels]
        kts_experts = [context_pos_kts_expert, context_neg_kts_expert, next_uttr_pos_kts_expert, next_uttr_neg_kts_expert]
        kts_task_loss = 0.0
        for (idx, expert), labels in zip(enumerate(kts_experts), kts_labels):
            expert = expert.unsqueeze(1).repeat(1, labels.size(1), 1)
            expert_scores = self.kts_prediction[idx](expert)
            expert_loss = self.compute_multi_task_loss(expert_scores, labels)
            label_masks = torch.sign(labels).float()
            expert_loss = torch.sum(expert_loss * label_masks.flatten()) / (torch.sum(label_masks.flatten(0)) + 1e-20)
            kts_task_loss += expert_loss
        return kts_task_loss

    def merge_keyword_info(
        self,
        data,
        context_pos_kts_expert, 
        context_neg_kts_expert, 
        next_uttr_pos_kts_expert, 
        next_uttr_neg_kts_expert
        ):
        context_infer_pos_kts = data["context_infer_pos_kts"]
        context_infer_neg_kts = data["context_infer_neg_kts"]
        next_uttr_infer_pos_kts = data["next_uttr_infer_pos_kts"]
        next_uttr_infer_neg_kts = data["next_uttr_infer_neg_kts"]
        infered_kts = [context_infer_pos_kts, context_infer_neg_kts, next_uttr_infer_pos_kts, next_uttr_infer_neg_kts]
        kts_experts = [context_pos_kts_expert, context_neg_kts_expert, next_uttr_pos_kts_expert, next_uttr_neg_kts_expert]
        merge_kts_experts = list()
        for expert, kts, attention, merge in zip(kts_experts, infered_kts, self.kts_attention, self.kts_merge):
            kts_emb = self.llama3_1.model.encoder.embed_tokens(kts) * self.llama3_1.model.encoder.embed_scale
            weighted_encoding, attention_weight = attention(
                query = expert[:,0,:].unsqueeze(1),
                memory = self.tanh(kts_emb),
                mask = kts.data.eq(self.tokenizer.pad_token_id)
            )
            kts_info = weighted_encoding.repeat(1, expert.size(1), 1)
            merge_kts_expert = merge(torch.cat((expert, kts_info), dim=-1))
            merge_kts_experts.append(merge_kts_expert)
        return merge_kts_experts

    def compute_multi_task_loss(self, scores, labels):
        labels = labels.view(-1)
        scores = scores.view(-1, scores.size(-1))
        loss = self.criterion_ce(scores, labels)
        return loss
    
    def select_experts(self, encoder_state, mixture_of_experts):
        zero_state = torch.zeros(encoder_state.size(), dtype=encoder_state.dtype).to(self.args.device)
        state_update = encoder_state
        bsz, num_experts, _, _ = mixture_of_experts.size() # bsz, num_experts, max_context_lengths, emb_dim
        expert_mask = torch.BoolTensor(np.ones((bsz, num_experts))).to(self.args.device)
        bsz_index = np.arange(bsz)
        sampled_experts_list = list()
        save_actions = list()
        save_action_probs = list()
        step = 0
        while step < self.args.max_num_actions:
            state = torch.cat([state_update] + [zero_state] * (self.args.max_num_actions-1-step), dim=-1)
            policy_outputs = self.policy_network(state)
            actor_logits = self.actor_network(policy_outputs)
            actor_logits[~expert_mask] = -float("inf")
            action_probs = F.softmax(actor_logits, dim=-1)
            state_values = self.critic_network(policy_outputs)
            m = Categorical(action_probs)
            sampled_action_idx = m.sample()
            # save sampled action probs
            sampled_action_probs = action_probs[bsz_index, sampled_action_idx.cpu().numpy().tolist()]
            save_action_probs.append(sampled_action_probs.unsqueeze(1))
            # update expert mask
            expert_mask[[bsz_index, sampled_action_idx.cpu().numpy().tolist()]] = 0 # bsz_idx is the index of dim=0, action_idx is the index of dim=1
            # save expert
            sampled_expert = mixture_of_experts[[bsz_index, sampled_action_idx.cpu().numpy().tolist()]]
            sampled_experts_list.append(sampled_expert)
            # save action probs, state values
            save_actions.append((m.log_prob(sampled_action_idx), state_values, m.entropy()))
            # update state
            state_update = torch.cat((state_update, sampled_expert[:,0,:]), dim=-1)
            step += 1
        return sampled_experts_list, save_actions, save_action_probs
    
    def get_actor_critic_loss(self, save_actions, batch_rewards):
        batch_rewards = torch.FloatTensor(batch_rewards).to(self.args.device)
        batch_rewards = batch_rewards.view(-1, 1).repeat(1, self.args.max_num_actions)
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(self.args.max_num_actions):
            log_prob, value, entropy = save_actions[i]
            advantage = batch_rewards[:,i] - value.squeeze(1)
            actor_loss += -log_prob * advantage.detach()
            critic_loss += advantage.pow(2)
            entropy_loss += entropy
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        ac_rl_loss = actor_loss + critic_loss + self.args.entropy_weight * entropy_loss
        return ac_rl_loss
    
    def get_self_critic_loss(self, save_action_probs):
        action_probs = torch.cat(save_action_probs, dim=1) # bsz, max_num_actions
        sc_rl_loss = torch.mean(torch.log(action_probs.flatten()))
        return sc_rl_loss
    
    def get_generator_rl_loss(self, response_outputs):
        response_scores = response_outputs.scores # list: max_length-1 (don't include __start__), (bsz, vocab) 
        response_scores = torch.cat([scores.unsqueeze(1) for scores in response_scores], dim=1) # bsz, max_length-1, vocab
        response_ids = response_outputs.sequences # bsz, max_length
        bsz, max_length =  response_ids.size()
        batch_output_logits = list()
        for idx in range(bsz):
            output_logits = response_scores[idx, range(max_length-1), response_ids[idx].cpu().numpy().tolist()[1:]]
            batch_output_logits.append(output_logits.unsqueeze(0))
        batch_output_logits = torch.cat(batch_output_logits, dim=0) # bsz, max_length-1
        batch_output_logits[torch.isinf(batch_output_logits) == True] = torch.tensor(0, dtype=torch.float, device=self.args.device) # for -inf
        batch_outputs = self.log_softmax(batch_output_logits)
        generator_rl_loss = - torch.mean(batch_outputs.flatten())
        return generator_rl_loss

    def get_rl_rewards(self, data, context, encoder_outputs, context_attention_mask, is_expert=False):
        paras = {}
        paras["encoder_outputs"] = (encoder_outputs, )
        paras["attention_mask"] = context_attention_mask
        response_outputs = self.llama3_1.generate(
            context,
            **paras, 
            max_length = 64,
            min_length = 5,
            num_beams = 1,
            pad_token_id = 0,
            use_cache = True,
            eos_token_id = self.tokenizer.eos_token_id, 
            temperature = 0.7,
            top_p = 0.9, 
            top_k = 30, 
            do_sample = True, 
            repetition_penalty = 1.03,
        )
        if is_expert:
            return response_outputs[:, 1:]
        response_results = [self.tokenizer.decode(response_outputs[:, :][idx].cpu(), skip_special_tokens=True) for idx in range(response_outputs.size(0))]
        # get rewards
        rewards, batch_rewards, eval_rewards = self.reward_agent.get_rewards(data, response_results)
        return rewards, batch_rewards, eval_rewards, response_outputs[:, 1:], response_results

    def rl_agent(self, data, encoder_state, mixture_of_experts):
        '''
        encoder_state: bsz, emb_dim
        mixture_of_experts: bsz, num_of_experts, max_seq_length, emb_dim
        '''
        context = data["context"]
        target = data["target"]
        target_role = data["target_role"]
        target_labels = data["target_lm_labels"]
        context_attention_mask = context.ne(self.tokenizer.pad_token_id)
        zero_state = torch.zeros(encoder_state.size(), dtype=encoder_state.dtype).to(self.args.device)
        state_update = encoder_state
        context_update = data["ori_context"]
        bsz_index = np.arange(encoder_state.size(0))
        save_actions = list()
        save_action_probs = list()
        save_rewards = list()
        negotiation_expert_loss_list =  list()
        politeness_expert_loss_list = list()
        kts_expert_loss_list = list()
        # expert_gen_loss_list = list()
        expert_mse_loss_list = list()
        step = 0
        while step < self.args.max_num_actions:
            state = torch.cat([state_update] + [zero_state] * (self.args.max_num_actions-1-step), dim=-1)
            policy_outputs = self.policy_network(state)
            actor_logits = self.actor_network(policy_outputs)
            action_probs = F.softmax(actor_logits, dim=-1)
            state_values = self.critic_network(policy_outputs)
            m = Categorical(action_probs)
            sampled_action_idx = m.sample()
            # save sampled action probs
            sampled_action_probs = action_probs[bsz_index, sampled_action_idx.cpu().numpy().tolist()]
            save_action_probs.append(sampled_action_probs.unsqueeze(1))
            # obtain sampled experts
            sampled_expert = mixture_of_experts[bsz_index, sampled_action_idx.cpu().numpy().tolist()]
            # save action probs, state values
            save_actions.append((m.log_prob(sampled_action_idx), state_values, m.entropy()))
            # generate experts' response
            response_outputs = self.get_rl_rewards(data, context, sampled_expert, context_attention_mask, is_expert=True)
            # get reward
            context_list = list()
            for idx in range(context_update.size(0)):
                context_pair = list()
                if self.tokenizer.pad_token_id in context_update[idx]:
                    context_pair.append(context_update[idx][:context_update[idx].cpu().numpy().tolist().index(self.tokenizer.pad_token_id)].unsqueeze(0))
                else:
                    context_pair.append(context_update[idx].unsqueeze(0))
                if self.tokenizer.pad_token_id in response_outputs[idx]:
                    context_pair.append(response_outputs[idx][:response_outputs[idx].cpu().numpy().tolist().index(self.tokenizer.pad_token_id)].unsqueeze(0))
                else:
                    context_pair.append(response_outputs[idx].unsqueeze(0))
                context_list.append(torch.cat(context_pair, dim=-1).squeeze(0))
            context_update = pad_sequence(context_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            context = deepcopy(context_update)
            context_attention_mask = context_update.ne(self.tokenizer.pad_token_id)
            encoder_outputs = self.llama3_1.model.encoder(
                context_update,
                attention_mask = context_attention_mask
            )
            encoder_outputs_other = encoder_outputs[1:]
            encoder_outputs = encoder_outputs[0]
            rewards, batch_rewards, eval_rewards, response_outputs, response_results = self.get_rl_rewards(data, context_update, encoder_outputs, context_attention_mask)
            # save rewards
            save_rewards.append(torch.FloatTensor(batch_rewards).to(self.args.device).unsqueeze(1))
            # update state
            state_update = torch.cat((state_update, encoder_outputs[:,0,:]), dim=-1)
            # update mixture-of-experts
            mixture_of_experts, negotiation_expert_loss, politeness_expert_loss, kts_expert_loss, expert_mse_loss = self.multi_task_experts(
                data,
                encoder_outputs
            )
            negotiation_expert_loss_list.append(negotiation_expert_loss)
            politeness_expert_loss_list.append(politeness_expert_loss)
            kts_expert_loss_list.append(kts_expert_loss)
            expert_mse_loss_list.append(expert_mse_loss)
                
            step += 1
        return context_update, encoder_outputs, context_attention_mask, \
               save_actions, save_action_probs, save_rewards, \
               negotiation_expert_loss_list, politeness_expert_loss_list, kts_expert_loss_list, expert_mse_loss_list, \
               rewards, eval_rewards, response_results
    
    def multi_task_experts(self, data, encoder_outputs):
        # negotiation strategy classification
        [context_negotiation_expert, next_uttr_negotiation_expert] = [mlp(encoder_outputs) for mlp in self.negotiation_experts]
        negotiation_expert_loss = self.negotiation_strategy_classification_task(
            data, 
            context_negotiation_expert[:,0,:],
            next_uttr_negotiation_expert[:,0,:], 
        )

        # politeness classification
        [context_pos_politeness_expert, context_neg_politeness_expert, next_uttr_pos_politeness_expert, next_uttr_neg_politeness_expert] = [mlp(encoder_outputs) for mlp in self.politeness_experts]
        politeness_expert_loss = self.politeness_classification_task(
            data, 
            context_pos_politeness_expert[:,0,:],
            context_neg_politeness_expert[:,0,:], 
            next_uttr_pos_politeness_expert[:,0,:], 
            next_uttr_neg_politeness_expert[:,0,:]
        )
        # keywords prediction
        [context_pos_kts_expert, context_neg_kts_expert, next_uttr_pos_kts_expert, next_uttr_neg_kts_expert] = [mlp(encoder_outputs) for mlp in self.kts_experts]
        [context_pos_kts_expert, context_neg_kts_expert, next_uttr_pos_kts_expert, next_uttr_neg_kts_expert] = self.merge_keyword_info(
            data,
            context_pos_kts_expert,
            context_neg_kts_expert,
            next_uttr_pos_kts_expert,
            next_uttr_neg_kts_expert
        )
        kts_expert_loss = self.keyword_prediction_task(
            data,
            context_pos_kts_expert[:,0,:],
            context_neg_kts_expert[:,0,:],
            next_uttr_pos_kts_expert[:,0,:],
            next_uttr_neg_kts_expert[:,0,:]
        )
        # Action
        mixture_of_experts = torch.cat((
            context_negotiation_expert.unsqueeze(1),
            next_uttr_negotiation_expert.unsqueeze(1),
            context_pos_politeness_expert.unsqueeze(1), 
            context_neg_politeness_expert.unsqueeze(1), 
            next_uttr_pos_politeness_expert.unsqueeze(1), 
            next_uttr_neg_politeness_expert.unsqueeze(1),
            context_pos_kts_expert.unsqueeze(1), 
            context_neg_kts_expert.unsqueeze(1), 
            next_uttr_pos_kts_expert.unsqueeze(1), 
            next_uttr_neg_kts_expert.unsqueeze(1)
        ), dim=1)
        # MSELoss
        expert_mse_loss = self.get_expert_mse_loss(mixture_of_experts, encoder_outputs)
        return mixture_of_experts, negotiation_expert_loss, politeness_expert_loss, kts_expert_loss, expert_mse_loss
    
    def get_rl_loss(self, save_actions, save_rewards):
        batch_rewards = torch.cat(save_rewards, dim=-1)
        num_steps = batch_rewards.size(1)
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.args.alpha * batch_rewards[:, num_steps - i]
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(num_steps):
            log_prob, value, entropy = save_actions[i]
            advantage = batch_rewards[:,i] - value.squeeze(1)
            actor_loss += -log_prob * advantage.detach()
            critic_loss += advantage.pow(2)
            entropy_loss += entropy
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        ac_rl_loss = actor_loss + critic_loss + self.args.entropy_weight * entropy_loss
        return ac_rl_loss

    def get_expert_mse_loss(self, mixture_of_experts, encoder_outputs):
        bsz, expert_num, _, _ = mixture_of_experts.size()
        expert_merge_rep = torch.mean(mixture_of_experts[:,:,0,:], dim=1) # bsz, emb
        encoder_rep = encoder_outputs[:, 0] # bsz, emb
        expert_mse_loss = self.args.expert_mse_weight * self.criterion_mse(expert_merge_rep, encoder_rep)
        return expert_mse_loss

    def forward(self, data, baseline_val=0.0, is_pretrain=False, is_test=False, is_train_rl=False, is_joint_train=False):
        context = data["context"]
        context_role = data["context_role"]
        context_token_type = data["context_token_type"]
        context_attention_mask = context.ne(self.tokenizer.pad_token_id)
        # Encode Context
        encoder_outputs = self.llama3_1.model.encoder(
            context,
            attention_mask = context_attention_mask,
            role_ids = context_role,
            turn_ids = context_token_type,
        )
        encoder_outputs_other = encoder_outputs[1:]
        encoder_outputs = encoder_outputs[0]

        # Experts-of-Multi-Tasks
        mixture_of_experts, negotiation_expert_loss, politeness_expert_loss, kts_expert_loss, expert_mse_loss = self.multi_task_experts(
            data,
            encoder_outputs
        )
        # bsz, _, seq_length, _ = mixture_of_experts.size()
        # encoder_outputs = self.expert_merge(mixture_of_experts.view(bsz, seq_length, -1))
        
        # RL Agent
        if not is_pretrain:
            context_update, encoder_outputs, context_attention_mask, \
            save_actions, save_action_probs, save_rewards, \
            negotiation_expert_loss_list, politeness_expert_loss_list, kts_expert_loss_list, expert_mse_loss_list, \
            rewards, eval_rewards, response_results = self.rl_agent(
                data, 
                encoder_outputs[:,0,:],
                mixture_of_experts
            )
            rl_agent_loss = self.get_rl_loss(save_actions, save_rewards)
            negotiation_expert_loss = torch.mean(torch.FloatTensor(negotiation_expert_loss_list+[negotiation_expert_loss]).to(self.args.device))
            politeness_expert_loss = torch.mean(torch.FloatTensor(politeness_expert_loss_list+[politeness_expert_loss]).to(self.args.device))
            kts_expert_loss = torch.mean(torch.FloatTensor(kts_expert_loss_list+[kts_expert_loss]).to(self.args.device))
            expert_mse_loss = torch.mean(torch.FloatTensor(expert_mse_loss_list+[expert_mse_loss]).to(self.args.device))
            #  expert_gen_loss = torch.mean(torch.FloatTensor(expert_gen_loss_list).to(self.args.device))

        # Decoding
        target = data["target"]
        target_role = data["target_role"]
        target_labels = data["target_lm_labels"]
        dec_outputs = self.llama3_1(
            encoder_outputs = (encoder_outputs, ),
            attention_mask = context_attention_mask,
            decoder_input_ids = target,
            decoder_role_ids = target_role,
            labels = target_labels
        )
        gen_loss = dec_outputs.loss
        ppl = torch.exp(gen_loss)
        
        if is_pretrain:
            loss = negotiation_expert_loss + politeness_expert_loss + kts_expert_loss + expert_mse_loss + gen_loss
            return_dict = {
                "loss": loss,
                "negotiation_expert_loss": negotiation_expert_loss,
                "politeness_expert_loss": politeness_expert_loss,
                "kts_expert_loss": kts_expert_loss,
                "expert_mse_loss": expert_mse_loss,
                "gen_loss": gen_loss,
                "ppl": ppl
            }
            return return_dict
        
        # Generation
        
        return_dict = {
            "negotiation_expert_loss": negotiation_expert_loss,
            "politeness_expert_loss": politeness_expert_loss,
            "kts_expert_loss": kts_expert_loss,
            "expert_mse_loss": expert_mse_loss,
            "gen_loss": gen_loss,
            "ppl": ppl,
            "weighted_rewards": rewards,
            "rewards": eval_rewards,
            "rl_loss": rl_agent_loss,
            # "expert_gen_loss": expert_gen_loss,
        }
        if is_train_rl:
            loss = negotiation_expert_loss + politeness_expert_loss + kts_expert_loss + expert_mse_loss + rl_agent_loss # + expert_gen_loss
            return_dict["loss"] = loss
            return return_dict
        
        if is_joint_train:
            loss = negotiation_expert_loss + politeness_expert_loss + kts_expert_loss + expert_mse_loss + rl_agent_loss + gen_loss # + expert_gen_loss
            return_dict["loss"] = loss
            return return_dict
        
        if is_test:
            loss = negotiation_expert_loss + politeness_expert_loss + kts_expert_loss + expert_mse_loss + rl_agent_loss + gen_loss # + expert_gen_loss
            return_dict["loss"] = loss
            return return_dict, response_results

