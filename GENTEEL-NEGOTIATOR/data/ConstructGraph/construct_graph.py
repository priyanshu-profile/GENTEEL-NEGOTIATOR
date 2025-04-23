#Negotiation and Politeness
from ast import keyword
from ntpath import join
import random
import pickle
import json
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from extraction import KeywordExtractor
from utils import simp_tokenize, kt_tokenize

class ConsConvGraph:
    def __init__(self, train_path, dev_path, test_path, conceptnet_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.conceptnet = json.load(open(conceptnet_path, "r", encoding="utf-8"))
        self.total_dialogs = self.get_dialogs(self.train_path)
        self.vocab = self.get_vocabulary()
        self.stop_words = stopwords.words("english")
        self.politeness_markers = self.extract_politeness_markers()

        # constrcut edge
        self.max_vertex = 10000
        self.min_outdegrees = 0
        self.min_pairs_num = 2
        self.Threshold = 0

    def get_dialogs(self, data_file):
        # total_data = json.load(open(data_file, encoding="utf-8"))
        total_data = pd.read_csv(data_file)
        total_dialogs = list()
        for session in total_data.groupby('dialogueId'):
            dialogs = defaultdict(list)
            buyer_uttr = []
            seller_uttr = []
            for i in range(len(session[1])):
                temp = session[1].iloc[i]
                dialog = temp.to_dict()
                #print('i,dialog',i,dialog)
                uttr = dialog["utterance"].strip()
                #print('uttr',uttr)
                if dialog["speaker"] == 0: #buyer utterance
                    #buyer_uttr.append(uttr)
                    buyer_uttr  = uttr
                    # if i < len(session)-1 and session["dialog"][i+1]["speaker"] == "seeker":
                    #     continue
                    # else:
                    #     dialogs["dialog"].append(" ".join(buyer_uttr))
                    dialogs["dialog"].append(buyer_uttr)
                    buyer_uttr = []
                elif dialog["speaker"] == 1:  # seller utterance # Consider supporter's multiple consecutive utterances as a buyer's response
                    #seller_uttr.append(uttr)
                    seller_uttr = uttr
                    # if i < len(session["dialog"])-1 and session["dialog"][i+1]["speaker"] == "supporter":
                    #     continue
                    # else:
                    #     dialogs["dialog"].append(" ".join(seller_uttr))
                    dialogs["dialog"].append(seller_uttr)
                    seller_uttr = []
            # print('dialogs',dialogs)
            # exit()
            total_dialogs.append(dialogs)
        #print('total_dialogs1',total_dialogs[0])
        

        return total_dialogs
    
    def get_vocabulary(self):
        vocab_counter = Counter()
        for session in self.total_dialogs:
            for uttr in session["dialog"]:
                vocab_counter.update(simp_tokenize(uttr))
        print("total vocabulary count: ", len(vocab_counter.items()))
        vocab = [token for token, _ in sorted(list(vocab_counter.items()), key=lambda x: (-x[1], x[0]))]
        return vocab
    
    def calculate_idf(self):
        word_counter = Counter()
        total_uttr = 0
        for session in self.total_dialogs:
            for uttr in session["dialog"]:
                word_counter.update(set(kt_tokenize(uttr)))
            total_uttr += len(session["dialog"])
        idf_dict = {}
        for kt, times in word_counter.items():
            idf_dict[kt] = np.log10(total_uttr / (times+1.))
        return idf_dict
    
    def extract_keywords(self):
        idf_dict = self.calculate_idf()
        keyword_extractor = KeywordExtractor(idf_dict)
        for session in tqdm(self.total_dialogs, desc="Extract Keywords"):
            session["ktlist"] = []
            for dialog in session["dialog"]:
                session["ktlist"].append(keyword_extractor.idf_extract(dialog, self.stop_words))

    def extract_politeness_markers(self):
        # idf_dict = self.calculate_idf()
        marker_extractor = KeywordExtractor(None)
        for session in tqdm(self.total_dialogs, desc="Extract Markers"):
            session["ktlist"] = []
            for dialog in session["dialog"]:
                positive_markers = []
                negative_markers = []
                positive_strategy = []
                negative_strategy = []
                politeness_strategy_marker = marker_extractor.politeness_marker_extract(dialog)
                #print('politeness_strategy_marker',politeness_strategy_marker)
                for marker_list in politeness_strategy_marker['positive'].values():
                    #print('marker_list',marker_list)
                    positive_markers.append(marker_list)
                    positive_strategy.append(['positive']*len(marker_list))
                

                for marker_list in politeness_strategy_marker['negative'].values():
                    #print('marker_list2',marker_list)
                    negative_markers.append(marker_list)
                    negative_strategy.append(['negative']*len(marker_list))
                one_marker_list = [item for sublist in positive_markers+negative_markers for item in sublist]
                one_strategy_list = [item for sublist in positive_strategy+negative_strategy for item in sublist]
                #print('positive_markers+negative_markers',one_dim_list)
                session["ktlist"].append(one_marker_list)
                session['ktstrategy'].append(one_strategy_list)
    
    def construct_forward_edge(self):
        '''
        Edges derive from two construction methods:
        - PMI
        - ConceptNet (limited number of hops, default 2-hop) (give up)
        '''
        #self.extract_keywords()
        self.extract_politeness_markers()
        word_dict = defaultdict() # Record the number of occurrences of each keyword
        co_occurrence = defaultdict(dict) # Record the number of occurrences of each keyword-pair
        occurrence_p = defaultdict()
        co_occurrence_p = defaultdict()
        ori_conv_graph = defaultdict(dict)

        # print('total_dialogs',self.total_dialogs[0])
        # exit()
        # for word in self.vocab:
        #     if len(word) > 1 and word not in co_occurrence.keys():
        #         word_dict[word] = 0.0   
        #         co_occurrence[word] = defaultdict()  
        for session in self.total_dialogs:
            for idx in range(len(session["ktlist"])-1):
                current_ktlist = session["ktlist"][idx]
                next_ktlist = session["ktlist"][idx+1]
                for kts in current_ktlist:
                    if len(kts) <= 2 or "\'" in kts or "=" in kts or "." in kts or kts not in self.conceptnet:
                        continue
                    if kts not in word_dict:
                        word_dict[kts] = 1.0
                    else:
                        word_dict[kts] += 1.0
                for current_kts in current_ktlist:
                    if len(current_kts) <= 2 or "\'" in current_kts or "=" in current_kts or "." in current_kts or current_kts not in self.conceptnet:
                        continue
                    for next_kts in next_ktlist:
                        # if next_kts == current_kts:
                        #     continue
                        if len(next_kts) <= 2 or "\'" in next_kts or "=" in next_kts or "." in next_kts or next_kts not in self.conceptnet:
                            continue
                        if next_kts not in co_occurrence[current_kts]:
                            co_occurrence[current_kts][next_kts] = 1.0
                        else:
                            co_occurrence[current_kts][next_kts] += 1.0
                if idx == len(session["ktlist"])-2:
                    for kts in next_ktlist:
                        if len(kts) <= 2 or "\'" in kts or "=" in kts or "." in kts or kts not in self.conceptnet:
                            continue
                        if kts not in word_dict:
                            word_dict[kts] = 1.0
                        else:
                            word_dict[kts] += 1.0
        
        occurrence_num = sum(word_dict.values())
        for word, times in word_dict.items():
            occurrence_p[word] = times / occurrence_num
        
        for word, co_occurrence_word_dict in co_occurrence.items():
            co_occurrence_p[word] = {}
            co_occurrence_sum = sum(co_occurrence_word_dict.values())
            size = len(co_occurrence_word_dict)
            if size < self.min_outdegrees:
                continue
            for co_occurrence_word, co_occurrence_num in co_occurrence_word_dict.items():
                if co_occurrence_num < self.min_pairs_num:
                    continue
                co_occurrence_p[word][co_occurrence_word] = co_occurrence_num / co_occurrence_sum
                co_occurrence_p[word][co_occurrence_word] = np.log(co_occurrence_p[word][co_occurrence_word] / occurrence_p[co_occurrence_word])
            if len(co_occurrence_p[word]) == 0:
                continue
            else:
                ori_conv_graph[word] = {}
            sort_words = list(sorted(co_occurrence_p[word].items(), key=lambda x: x[1], reverse=True))
            max_num = self.max_vertex if size > self.max_vertex else size
            for sort_word, sort_pmi in sort_words[:max_num]:
                if sort_pmi > self.Threshold:
                    ori_conv_graph[word][sort_word] = sort_pmi
        
        with open("./dialog.json","w",encoding="utf-8") as f:
            json.dump(self.total_dialogs, f, ensure_ascii=False, indent=2)
        
        print("Dialog Done!") 

        with open("./forward_ori_conv_graph.json","w",encoding="utf-8") as f:
            json.dump(ori_conv_graph, f, ensure_ascii=False, indent=2)
        
        print("Extract Keyword and Construct Forward Original Graph Done!")
        return ori_conv_graph
    
    def construct_backtard_edge(self):
        '''
        Edges derive from two construction methods:
        - PMI
        - ConceptNet (limited number of hops, default 2-hop) (give up)
        '''
        word_dict = defaultdict() # Record the number of occurrences of each keyword
        co_occurrence = defaultdict(dict) # Record the number of occurrences of each keyword-pair
        occurrence_p = defaultdict()
        co_occurrence_p = defaultdict()
        ori_conv_graph = defaultdict(dict)
        for session in self.total_dialogs:
            for idx in range(len(session["ktlist"])-1, 0, -1):
                current_ktlist = session["ktlist"][idx]
                next_ktlist = session["ktlist"][idx-1]
                for kts in current_ktlist:
                    if len(kts) <= 2 or "\'" in kts or "=" in kts or "." in kts or kts not in self.conceptnet:
                        continue
                    if kts not in word_dict:
                        word_dict[kts] = 1.0
                    else:
                        word_dict[kts] += 1.0
                for current_kts in current_ktlist:
                    if len(current_kts) <= 2 or "\'" in current_kts or "=" in current_kts or "." in current_kts or current_kts not in self.conceptnet:
                        continue
                    for next_kts in next_ktlist:
                        # if next_kts == current_kts:
                        #     continue
                        if len(next_kts) <= 2 or "\'" in next_kts or "=" in next_kts or "." in next_kts or next_kts not in self.conceptnet:
                            continue
                        if next_kts not in co_occurrence[current_kts]:
                            co_occurrence[current_kts][next_kts] = 1.0
                        else:
                            co_occurrence[current_kts][next_kts] += 1.0
                if idx == 1:
                    for kts in next_ktlist:
                        if len(kts) <= 2 or "\'" in kts or "=" in kts or "." in kts or kts not in self.conceptnet:
                            continue
                        if kts not in word_dict:
                            word_dict[kts] = 1.0
                        else:
                            word_dict[kts] += 1.0
        
        occurrence_num = sum(word_dict.values())
        for word, times in word_dict.items():
            occurrence_p[word] = times / occurrence_num
        
        for word, co_occurrence_word_dict in co_occurrence.items():
            co_occurrence_p[word] = {}
            co_occurrence_sum = sum(co_occurrence_word_dict.values())
            size = len(co_occurrence_word_dict)
            if size < self.min_outdegrees:
                continue
            for co_occurrence_word, co_occurrence_num in co_occurrence_word_dict.items():
                if co_occurrence_num < self.min_pairs_num:
                    continue
                co_occurrence_p[word][co_occurrence_word] = co_occurrence_num / co_occurrence_sum
                co_occurrence_p[word][co_occurrence_word] = np.log(co_occurrence_p[word][co_occurrence_word] / occurrence_p[co_occurrence_word])
            if len(co_occurrence_p[word]) == 0:
                continue
            else:
                ori_conv_graph[word] = {}
            sort_words = list(sorted(co_occurrence_p[word].items(), key=lambda x: x[1], reverse=True))
            max_num = self.max_vertex if size > self.max_vertex else size
            for sort_word, sort_pmi in sort_words[:max_num]:
                if sort_pmi > self.Threshold:
                    ori_conv_graph[word][sort_word] = sort_pmi
        
        with open("./backtard_ori_conv_graph.json","w",encoding="utf-8") as f:
            json.dump(ori_conv_graph, f, ensure_ascii=False, indent=2)
        
        print("Extract Keyword and Construct Backtard Original Graph Done!")
        return ori_conv_graph

    # def construct_conv_graph(self, forward_ori_conv_graph, backtard_ori_conv_graph, vad_dict):
    def construct_conv_graph(self, forward_ori_conv_graph, backtard_ori_conv_graph):
        '''
        the form of graph: {head_vertex : [(forward, positive, tail_vertex), (backtard, negative, tail_vertex),...]}
        '''
        
        forward_ori_conv_graph = forward_ori_conv_graph
        backtard_ori_conv_graph = backtard_ori_conv_graph
        conv_graph = defaultdict(list)
        for session in tqdm(self.total_dialogs, desc="Construct Conv Graph: "):
            # print(session)
            for idx in range(len(session["ktlist"])-1):
                current_ktlist = session["ktlist"][idx]
                next_ktlist = session["ktlist"][idx+1]
                current_ktstrategylist = session["ktstrategy"][idx]
                next_ktstrategylist = session["ktstrategy"][idx+1]

                # current_ktlist = ['hey', 'i', 'my', 'what'] 
                # next_ktlist = ['i', 'your', 'you', 'you']
                # current_ktstrategylist = ['positive', 'negative', 'negative', 'negative']
                # next_ktstrategylist = ['negative', 'negative', 'negative', 'negative']

                # forward view

                # print('current_ktlist',current_ktlist)
                # print('next_ktlist',next_ktlist)
                # print('current_ktstrategy',current_ktstrategylist)
                # print('next_ktstrategy',next_ktstrategylist)

                for i, current_kts in enumerate(current_ktlist):
                    # current_kts_valence = vad_dict[current_kts][0] if current_kts in vad_dict else 0.5
                    current_kts_politeness = current_ktstrategylist[i]
                    #print('current_kts',current_kts, current_kts_politeness)
                    for j, next_kts in enumerate(next_ktlist):
                        if current_kts not in forward_ori_conv_graph:
                            continue
                        if next_kts not in forward_ori_conv_graph[current_kts].keys():
                            continue
                        if current_kts in conv_graph:
                            neighbors = [(direct, kt) for direct, _, kt, _ in conv_graph[current_kts]]
                            if ("forward", next_kts) in neighbors:
                                continue
                        # next_kts_valence = vad_dict[next_kts][0] if next_kts in vad_dict else 0.5
                        next_kts_politeness = next_ktstrategylist[j]
                        
                        # forward
                        # if next_kts_valence >= 0.5:
                        if next_kts_politeness == 'positive':
                            conv_graph[current_kts].append(("forward", "positive", next_kts, forward_ori_conv_graph[current_kts][next_kts]))
                        else:
                            conv_graph[current_kts].append(("forward", "negative", next_kts, forward_ori_conv_graph[current_kts][next_kts]))
                        # backtard
                        # if current_kts_valence >= 0.5:
                        #     conv_graph[next_kts].append(("backtard", "positive", current_kts))
                        # else:
                        #     conv_graph[next_kts].append(("backtard", "negative", current_kts))
                
                # backtard view
                for k, next_kts in enumerate(next_ktlist):
                    # next_kts_valence = vad_dict[next_kts][0] if next_kts in vad_dict else 0.5
                    next_kts_politeness = next_ktstrategylist[k]
                    for l, current_kts in enumerate(current_ktlist):
                        if next_kts not in backtard_ori_conv_graph:
                            continue
                        if current_kts not in backtard_ori_conv_graph[next_kts].keys():
                            continue
                        if next_kts in conv_graph:
                            neighbors = [(direct, kt) for direct, _, kt, _ in conv_graph[next_kts]]
                            if ("backtard", current_kts) in neighbors:
                                continue
                        # current_kts_valence = vad_dict[current_kts][0] if current_kts in vad_dict else 0.5
                        current_kts_politeness = current_ktstrategylist[l]
                        # backtard
                        # if current_kts_valence >= 0.5:
                        if current_kts_politeness == 'positive':
                            conv_graph[next_kts].append(("backtard", "positive", current_kts, backtard_ori_conv_graph[next_kts][current_kts]))
                        else:
                            conv_graph[next_kts].append(("backtard", "negative", current_kts, backtard_ori_conv_graph[next_kts][current_kts]))
                        # forward
                        # if next_kts >= 0.5:
                        #     conv_graph[current_kts].append(("forward", "positive", next_kts))
                        # else:
                        #     conv_graph[current_kts].append(("forward", "negative", next_kts))
        with open("./conv_graph.json", "w", encoding="utf-8") as f:
            json.dump(conv_graph, f, ensure_ascii=False, indent=2)
        print("Conversational Graph Constructed Done!")

        total_kts = list()
        for kt, neighbors in conv_graph.items():
            total_kts.append(kt)
            for _, _, tail, _ in neighbors:
                total_kts.append(tail)
        total_kts = list(set(total_kts))
        with open("./total_kts.pkl", "wb") as f:
            pickle.dump(total_kts, f)

def main():
    train_path = "../train_conv.csv"
    dev_path = "../dev_conv.csv"
    test_path = "../test_conv.csv"
    conceptnet_path = "./ConceptNet.json"
    graph = ConsConvGraph( train_path, dev_path, test_path, conceptnet_path)
    forward_ori_conv_graph = graph.construct_forward_edge()
    backtard_ori_conv_graph = graph.construct_backtard_edge()
    # vad_dict = json.load(open(f"../VAD.json", "r", encoding="utf-8"))
    # graph.construct_conv_graph(forward_ori_conv_graph, backtard_ori_conv_graph, vad_dict)
    graph.construct_conv_graph(forward_ori_conv_graph, backtard_ori_conv_graph)

if __name__ == "__main__":
    main()