import os
import json
import warnings
warnings.filterwarnings('ignore')
import nltk
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from utils import kt_tokenize
from constants import WORD_PAIRS as word_pairs
stopwords = stopwords.words("english")

class ConstructSimulatorData:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.esconv_file = args.esconv
        self.conv_graph = json.load(open(args.conv_graph, "r", encoding="utf-8"))
        self.politeness_dict = json.load(open(args.politeness, "r", encoding="utf-8"))
        self.total_kts = pickle.load(open(args.total_kts, "rb"))
        self.politeness_statistic = json.load(open(args.politeness_statistic, "r", encoding="utf-8"))
        self.max_num_dialog = args.max_num_dialog
        self.max_num_kts = args.max_num_kts
        self.args = args
        
    def split_dataset(self):
        dataset = json.load(open(self.esconv_file, encoding="utf-8"))
        print("Num of Original Dialog Session: ", len(dataset))
        
        dev_file = f"{self.data_dir}/dev_data.json"
        test_file = f"{self.data_dir}/test_data.json"
        train_file = f"{self.data_dir}/train_data.json"
        if not os.path.exists(train_file) or not os.path.exists(dev_file) or not os.path.exists(test_file):
            # Split Train/dev/Test
            print("Split Train/Dev/Test......")
            total_data = []
            for session in tqdm(dataset):
                politeness = session["politeness"]
                dialog = session["dialog"]
                context = []
                context_role = []
                context_strategy_seqs = []
                context_positive_kts = []
                context_negative_kts = []
                context_infer_kts = []
                context_politeness_scores = []
                dialog_turn = 0
                context_user_sum_pol_score = 0.0
                user_content = []
                for idx, info in enumerate(dialog):
                    speaker = info["speaker"]
                    if speaker == "user":
                        content = info["content"]
                        user_content.append(content.strip())
                        if idx < len(dialog) - 1 and dialog[idx + 1]["speaker"] == "user":
                            continue
                        else:
                            content = " ".join(user_content)
                            context.append(content)
                            user_content = []
                            dialog_turn += 1
                            context_role.append(speaker)
                            context_strategy_seqs.append("none")
                            con_pos_kts, con_neg_kts, con_inf_kts = self.extract_keyterms(content.strip(), "forward")
                            context_positive_kts.append(con_pos_kts)
                            context_negative_kts.append(con_neg_kts)
                            context_infer_kts.append(con_inf_kts)
                            con_pol_score = self.get_politeness_score(content.strip())
                            context_politeness_scores.append(con_pol_score)
                            context_user_sum_pol_score += con_pol_score
                    elif speaker == "agent":
                        assert "strategy" in info["annotation"]
                        strategy = info["annotation"]["strategy"]
                        target = info["content"]
                        tar_pos_kts, tar_neg_kts, _ = self.extract_keyterms(target.strip())
                        tar_pol_score = self.get_politeness_score(target.strip())
                        if "user" in context_role[-self.max_num_dialog:]:
                            next_uttr_idx = idx+1
                            next_uttr = ""
                            next_uttr_list = []
                            next_uttr_pos_kts = []
                            next_uttr_neg_kts = []
                            next_uttr_infer_kts = []
                            user_flag = False
                            for next_idx in range(next_uttr_idx, len(dialog)):
                                if dialog[next_idx]["speaker"] == "user":
                                    next_uttr_list.append(dialog[next_idx]["content"].strip())
                                    user_flag = True
                                    if next_idx < len(dialog)-1 and dialog[next_idx+1]["speaker"] == "user":
                                        continue
                                elif dialog[next_idx]["speaker"] == "agent":
                                    if user_flag:
                                        break
                            next_uttr = " ".join(next_uttr_list).strip()
                            next_uttr_pos_kts, next_uttr_neg_kts, next_uttr_infer_kts = self.extract_keyterms(next_uttr, "backtard")
                            next_uttr_pol_score = self.get_politeness_score(next_uttr)
                            save_data = {
                                "politeness_type": politeness_type,
                                "strategy": strategy,
                                "context_strategy_seqs": deepcopy(context_strategy_seqs[-self.max_num_dialog:]),
                                "context_role": deepcopy(context_role[-self.max_num_dialog:]),
                                "dialog_turn": dialog_turn,
                                "context": deepcopy(context[-self.max_num_dialog:]),
                                "context_positive_kts": deepcopy(context_positive_kts[-self.max_num_dialog:]),
                                "context_negative_kts": deepcopy(context_negative_kts[-self.max_num_dialog:]),
                                "context_infer_kts": deepcopy(context_infer_kts[-self.max_num_dialog:]),
                                "context_politeness_scores": deepcopy(context_politeness_scores[-self.max_num_dialog:]),
                                "context_user_sum_pol_score": context_user_sum_pol_score,
                                "target": target,
                                "target_positive_kts": tar_pos_kts,
                                "target_negative_kts": tar_neg_kts,
                                "target_politeness_score": tar_pol_score,
                                "next_uttr": next_uttr,
                                "next_uttr_positive_kts": next_uttr_pos_kts,
                                "next_uttr_negative_kts": next_uttr_neg_kts,
                                "next_uttr_infer_kts": next_uttr_infer_kts,
                                "next_uttr_politeness_score": next_uttr_pol_score,
                            }
                            total_data.append(save_data)
                        context.append(target)
                        context_role.append(speaker)
                        context_strategy_seqs.append(strategy)
                        context_positive_kts.append(tar_pos_kts)
                        context_negative_kts.append(tar_neg_kts)
                        context_infer_kts.append([])
                        context_politeness_scores.append(tar_pol_score)
                    else:
                        raise ValueError(speaker + " Wrong!")
                assert len(context) == len(context_positive_kts) == len(context_negative_kts) == len(context_role) == len(context_strategy_seqs) == len(context_infer_kts) == len(context_politeness_scores)
            random.seed(self.args.seed)
            random.shuffle(total_data)
            dev_size = int(0.1 * len(total_data))
            test_size = int(0.1 * len(total_data))
            dev_data = total_data[:dev_size]
            test_data = total_data[dev_size: dev_size+test_size]
            train_data = total_data[dev_size+test_size:]
            with open(train_file,"w",encoding="utf-8") as f:
                json.dump(train_data, f, indent=2)
            with open(dev_file,"w",encoding="utf-8") as f:
                json.dump(dev_data, f, indent=2)
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2)
            print("Total Data: ", len(total_data))
            print("Train Data: ", len(train_data))
            print("Dev Data: ", len(dev_data))
            print("Test Data: ", len(test_data))
            print("Split Complete!")
        else:
            train_data = json.load(open(train_file,"r",encoding="utf-8"))
            dev_data = json.load(open(dev_file,"r",encoding="utf-8"))
            test_data = json.load(open(test_file, "r", encoding="utf-8"))
    
        return train_data, dev_data, test_data


    def _norm(self, x):
        return ' '.join(x.strip().split())

    def process_sent(self, sentence):
        sentence = sentence.lower()
        for k, v in word_pairs.items():
            sentence = sentence.replace(k, v)
        sentence = nltk.word_tokenize(sentence)
        return sentence


    def infer_politeness(self, data):

        data_dict = {
            "context_role": [],
            "dialog_turn": [],
            "context": [],
            "context_strategy_seqs": [],
            "context_positive_kts": [],
            "context_negative_kts": [],
            "context_last_pos_kts": [],
            "context_last_neg_kts": [],
            "context_last_infer_kts": [],
            "context_politeness_scores": [],
            "context_user_sum_pol_score": [],
            "strategy": [],
            "target": [],
            "target_positive_kts": [],
            "target_negative_kts": [],
            "next_uttr": [],
            "next_uttr_positive_kts": [],
            "next_uttr_negative_kts": [],
            "next_uttr_infer_kts": [],
            "next_uttr_politeness_score": [],
            "politeness": [],
            "politeness_context": [],
            "utt_cs": [],
            "next_uttr_cs": [],
            "situation_cs": [],
            "target_cs": [],
        }
        comet = Comet(self.args.comet_file, self.args.device)
        for line in tqdm(data):
            dialog_turn = line["dialog_turn"]
            context_role = line["context_role"] # list
            context_strategy_seqs = line["context_strategy_seqs"]
            context_positive_kts = line["context_positive_kts"]
            context_negative_kts = line["context_negative_kts"]
            context_infer_kts = line["context_infer_kts"]
            context_politeness_scores = line["context_politeness_scores"]
            context_user_sum_pol_score = line["context_user_sum_pol_score"]
            politeness_type = line["politeness_type"] # str
            target = self.process_sent(self._norm(line["target"]))
            target_positive_kts = line["target_positive_kts"]
            target_negative_kts = line["target_negative_kts"]
            next_uttr = self.process_sent(self._norm(line["next_uttr"]))
            next_uttr_positive_kts = line["next_uttr_positive_kts"]
            next_uttr_negative_kts = line["next_uttr_negative_kts"]
            next_uttr_infer_kts = line["next_uttr_infer_kts"]
            next_uttr_politeness_score = line["next_uttr_politeness_score"]
            strategy = line["strategy"]
            context_list = []
            politeness_list = []
            for uttr in line["context"]:
                item = self.process_sent(self._norm(uttr))
                context_list.append(item)
            last_user_idx = max(index for index, role in enumerate(context_role) if role == "user")
            context_last_pos_kts = context_positive_kts[last_user_idx]
            context_last_neg_kts = context_negative_kts[last_user_idx]
            context_last_infer_kts = context_infer_kts[last_user_idx]
            data_dict["dialog_turn"].append(dialog_turn)
            data_dict["context"].append(context_list)
            data_dict["context_role"].append(context_role)
            data_dict["context_strategy_seqs"].append(context_strategy_seqs)
            data_dict["context_positive_kts"].append(context_positive_kts)
            data_dict["context_negative_kts"].append(context_negative_kts)
            data_dict["context_last_pos_kts"].append(context_last_pos_kts)
            data_dict["context_last_neg_kts"].append(context_last_neg_kts)
            data_dict["context_last_infer_kts"].append(context_last_infer_kts)
            data_dict["context_politeness_scores"].append(context_politeness_scores)
            data_dict["context_user_sum_pol_score"].append(context_user_sum_pol_score)
            data_dict["politeness"].append(politeness_type)
            data_dict["politeness_context"].append(politeness_list)
            data_dict["strategy"].append(strategy)
            data_dict["target"].append(target)
            data_dict["target_positive_kts"].append(target_positive_kts)
            data_dict["target_negative_kts"].append(target_negative_kts)
            data_dict["next_uttr"].append(next_uttr)
            data_dict["next_uttr_positive_kts"].append(next_uttr_positive_kts)
            data_dict["next_uttr_negative_kts"].append(next_uttr_negative_kts)
            data_dict["next_uttr_infer_kts"].append(next_uttr_infer_kts)
            data_dict["next_uttr_politeness_score"].append(next_uttr_politeness_score)
            data_dict["utt_cs"].append(utt_cs)
            data_dict["next_uttr_cs"].append(next_uttr_cs)
            data_dict["target_cs"].append(target_cs)
           
        assert len(data_dict["context"])==len(data_dict["context_role"])==len(data_dict["situation"])==len(data_dict["politeness"])==len(data_dict["politeness_context"])==len(data_dict["target"])==len(data_dict["utt_cs"])==len(data_dict["situation_cs"])==len(data_dict["context_strategy_seqs"])==len(data_dict["strategy"])==len(data_dict["context_positive_kts"])==len(data_dict["context_negative_kts"])==len(data_dict["target_negative_kts"])==len(data_dict["target_positive_kts"])==len(data_dict["context_last_pos_kts"])==len(data_dict["context_last_neg_kts"])==len(data_dict["next_uttr"])==len(data_dict["next_uttr_cs"])==len(data_dict["next_uttr_positive_kts"])==len(data_dict["next_uttr_negative_kts"])==len(data_dict["dialog_turn"])==len(data_dict["context_last_infer_kts"])==len(data_dict["next_uttr_infer_kts"])==len(data_dict["target_cs"])==len(data_dict["context_politeness_scores"])==len(data_dict["context_user_sum_pol_score"])==len(data_dict["next_uttr_politeness_score"])
        return data_dict
        
    def extract_keyterms(self, utterance, relation=None):
        positive_kts = list()
        negative_kts = list()
        infer_kts = list()
        uttr_kts = kt_tokenize(utterance)
        for kts in uttr_kts:
            if kts not in self.total_kts:
                continue
            kts_politeness = self.politeness_dict[kts][0] if kts in self.politeness_dict else 0.5
            if kts_politeness >= 0.5:
                positive_kts.append(kts)
            else:
                negative_kts.append(kts)
            if relation is not None:
                kts_neighbors = [neighbor for infer_relation,_,neighbor,_ in self.conv_graph[kts] if infer_relation==relation]
                infer_kts.extend(kts_neighbors)
        return deepcopy(list(set(positive_kts))), deepcopy(list(set(negative_kts))), deepcopy(list(set(infer_kts)))
    
    def get_politeness_score(self, utterance):
        politeness_score = list()
        uttr_words = kt_tokenize(utterance)
        for word in uttr_words:
            if word not in stopwords and word.isalpha(): # and word in self.politeness_dict:
                word_politeness = self.politeness_dict[word][0] if word in self.politeness_dict else 0.5
                politeness_score.append(word_politeness)
        avg_score = 0.5
        if len(politeness_score) > 0:
            avg_score = sum(politeness_score)/len(politeness_score)
        return avg_score
    
    def extract_labels(self, data):
        '''
        context_pos_pol_label, context_neg_pol_label
        next_uttr_pos_pol_label, next_uttr_neg_pol_label
        context_infer_pos_kts, context_infer_neg_kts
        next_uttr_infer_pos_kts, next_uttr_infer_neg_kts
        '''
        context_pos_pol_labels = list()
        context_neg_pol_labels = list()
        next_uttr_pos_pol_labels = list()
        next_uttr_neg_pol_labels = list()
        # politeness labels
        for uttr_cs, next_uttr_cs in zip(data["utt_cs"], data["next_uttr_cs"]):
            # context
            context_pos_pol = list()
            context_neg_pol = list()
            for uttr_reaction in uttr_cs["xPol"]:
                if len(uttr_reaction) == 1:
                    uttr_pol = uttr_reaction[0]
                    if uttr_pol in self.politeness_statistic:
                        if self.politeness_statistic[uttr_pol]["politeness"] == "positive":
                            context_pos_pol.append(uttr_pol)
                        elif self.politeness_statistic[uttr_pol]["politeness"] == "negative":
                            context_neg_pol.append(uttr_pol)
                        else:
                            raise ValueError("Context Politeness Error!")
            context_pos_pol = list(set(context_pos_pol))
            context_neg_pol = list(set(context_neg_pol))
            context_pos_pol_labels.append(context_pos_pol)
            context_neg_pol_labels.append(context_neg_pol)
            # next uttr
            next_uttr_pos_pol = list()
            next_uttr_neg_pol = list()
            for next_uttr_reaction in next_uttr_cs["xPol"]:
                if len(next_uttr_reaction) == 1:
                    next_uttr_pol = next_uttr_reaction[0]
                    if next_uttr_pol in self.politeness_statistic:
                        if self.politeness_statistic[next_uttr_pol]["politeness"] == "positive":
                            next_uttr_pos_pol.append(next_uttr_pol)
                        elif self.politeness_statistic[next_uttr_pol]["politeness"] == "negative":
                            next_uttr_neg_pol.append(next_uttr_pol)
                        else:
                            raise ValueError("Next Uttr Politeness Error!")
            next_uttr_pos_pol = list(set(next_uttr_pos_pol))
            next_uttr_neg_pol = list(set(next_uttr_neg_pol))
            next_uttr_pos_pol_labels.append(next_uttr_pos_pol)
            next_uttr_neg_pol_labels.append(next_uttr_neg_pol)
        assert len(context_pos_pol_labels) == len(context_neg_pol_labels) == len(next_uttr_pos_pol_labels) == len(next_uttr_neg_pol_labels) == len(data["utt_cs"]) == len(data["next_uttr_cs"])
        data["context_pos_pol_labels"] = context_pos_pol_labels
        data["context_neg_pol_labels"] = context_neg_pol_labels
        data["next_uttr_pos_pol_labels"] = next_uttr_pos_pol_labels
        data["next_uttr_neg_pol_labels"] = next_uttr_neg_pol_labels

        # infer keyterms
        context_infer_pos_kts = list()
        context_infer_neg_kts = list()
        next_uttr_infer_pos_kts = list()
        next_uttr_infer_neg_kts = list()
        for context_last_pos_kts, context_last_neg_kts in tqdm(zip(data["context_last_pos_kts"], data["context_last_neg_kts"]), desc="Infer keyterms: "):
            context_last_kts = list(set(context_last_pos_kts + context_last_neg_kts))
            # infer kts from context: forward
            forward_pos_kts = list()
            forward_neg_kts = list()
            for context_kts in context_last_kts:
                for direction, polarity, kts, pmi in self.conv_graph[context_kts]:
                    if direction == "forward" and polarity == "positive":
                        forward_pos_kts.append((direction, polarity, kts, pmi))
                    elif direction == "forward" and polarity == "negative":
                        forward_neg_kts.append((direction, polarity, kts, pmi))
            forward_pos_kts = sorted(list(set(forward_pos_kts)), key=lambda x: x[-1], reverse=True)[:self.max_num_kts]
            forward_neg_kts = sorted(list(set(forward_neg_kts)), key=lambda x: x[-1], reverse=True)[:self.max_num_kts]
            context_infer_pos_kts.append(forward_pos_kts)
            context_infer_neg_kts.append(forward_neg_kts)
            
            # infer kts from next uttr: forward, forward, backtard
            next_uttr_kts = list()
            response_kts = list(set([kts for _,_,kts,_ in forward_pos_kts] + [kts for _,_,kts,_ in forward_neg_kts]))
            for res_kts in response_kts:
                for direction, _, kts, pmi in self.conv_graph[res_kts]:
                    if direction == "forward":
                        next_uttr_kts.append((kts, pmi))
            next_uttr_kts = sorted(list(set(next_uttr_kts)), key=lambda x: x[-1], reverse=True)[:self.max_num_kts]
            next_uttr_kts = list(set([kts for kts, _ in next_uttr_kts]))
            backtard_pos_kts = list()
            backtard_neg_kts = list()
            for next_kts in next_uttr_kts:
                for direction, polarity, kts, pmi in self.conv_graph[next_kts]:
                    if direction == "backtard" and polarity == "positive":
                        backtard_pos_kts.append((direction, polarity, kts, pmi))
                    elif direction == "backtard" and polarity == "negative":
                        backtard_neg_kts.append((direction, polarity, kts, pmi))
            backtard_pos_kts = sorted(list(set(backtard_pos_kts)), key=lambda x: x[-1], reverse=True)[:self.max_num_kts]
            backtard_neg_kts = sorted(list(set(backtard_neg_kts)), key=lambda x: x[-1], reverse=True)[:self.max_num_kts]
            next_uttr_infer_pos_kts.append(backtard_pos_kts)
            next_uttr_infer_neg_kts.append(backtard_neg_kts)
        assert len(context_infer_pos_kts) == len(context_infer_neg_kts) == len(next_uttr_infer_pos_kts) == len(next_uttr_infer_neg_kts) == len(data["context_last_pos_kts"]) == len(data["context_last_neg_kts"])
        data["context_infer_pos_kts"] = context_infer_pos_kts
        data["context_infer_neg_kts"] = context_infer_neg_kts
        data["next_uttr_infer_pos_kts"] = next_uttr_infer_pos_kts
        data["next_uttr_infer_neg_kts"] = next_uttr_infer_neg_kts
        
        return data

    def construct_dataset(self):
        dataset = f"{self.data_dir}/dataset_preproc.p"
        politeness_cache_file = f"{self.data_dir}/dataset_preproc_politeness_cache.p"

        # Split train/dev/test
        data_train, data_dev, data_test = self.split_dataset()

        # Infer positive/negative politeness labels
        if not os.path.exists(politeness_cache_file):
            data_train = self.infer_politeness(data_train)
            data_dev = self.infer_politeness(data_dev)
            data_test = self.infer_politeness(data_test)
            with open(politeness_cache_file, "wb") as f:
                pickle.dump([data_train, data_dev, data_test], f)
                print("Save politeness Cache File!")
        else:
            with open(politeness_cache_file, "rb") as f:
                [data_train, data_dev, data_test] = pickle.load(f)

        # Extract keyterm labels and politeness labels
        if not os.path.exists(dataset):
            data_train = self.extract_labels(data_train)
            data_dev = self.extract_labels(data_dev)
            data_test = self.extract_labels(data_test)
            with open(dataset, "wb") as f:
                pickle.dump([data_train, data_dev, data_test], f)
            print("Save Dataset File!")
        else:
            with open(dataset, "rb") as f:
                [data_train, data_dev, data_test] = pickle.load(f)

def main(args):
    random.seed(args.seed)
    dataset = ConstructDataset(args)
    dataset.construct_dataset()

def get_args():
    parser = argparse.ArgumentParser()
    
    # random seed
    parser.add_argument("--seed", type=int, default=13)

    # dataset
    parser.add_argument("--data_dir", type=str, default="..")
    parser.add_argument("--esconv", type=str, default="../ESConv.json")
    parser.add_argument("--politeness", type=str, default="../politeness.json")
    parser.add_argument("--max_num_dialog", type=int, default=10)
    parser.add_argument("--conv_graph", type=str, default="../ConstructConvGraph/conv_graph.json")
    parser.add_argument("--total_kts", type=str, default="../ConstructConvGraph/total_kts.pkl")
    parser.add_argument("--politeness_statistic", type=str, default="../politeness_statistic.json")
    sparser.add_argument("--max_num_kts", type=int, default=256)

    # gpu
    parser.add_argument("--gpu", type=int, default=3)

    args = parser.parse_args()
    cuda_id = "cuda:" + str(args.gpu)
    args.device = torch.device(cuda_id) if torch.cuda.is_available() else 'cpu'
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)