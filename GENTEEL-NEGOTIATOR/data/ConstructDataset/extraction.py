from utils import simp_tokenize, pos_tag, kt_tokenize
import pandas as pd
from convokit import PolitenessStrategies, Corpus, download, TextParser


class KeytermExtractor():
    def __init__(self, idf_dict=None):
        self.idf_dict = idf_dict

    @staticmethod
    def is_keyword_tag(tag):
        return tag.startswith('VB') or tag.startswith('NN') or tag.startswith('JJ')

    @staticmethod
    def cal_tag_score(tag):
        if tag.startswith('VB'):
            return 2
        if tag.startswith('NN'):
            return 2.
        if tag.startswith('JJ'):
            return 2
        return 0.

    def idf_extract(self, string, con_kt=None):
        tokens = simp_tokenize(string)
        seq_len = len(tokens)
        tokens = pos_tag(tokens)
        source = kt_tokenize(string)
        candi = []
        result = []
        for i, (word, tag) in enumerate(tokens):
            score = self.cal_tag_score(tag)
            # if not is_candiword(source[i]) or score == 0.:
            #     continue
            if score == 0.:
                continue
            if con_kt is not None and source[i] in con_kt:
                continue
            score *= source.count(source[i])
            score *= 1 / seq_len
            score *= self.idf_dict[source[i]]
            candi.append((source[i], score))
            if score > 0.1:
                result.append(source[i])
        return list(set(result))

    def extract(self, string):
        tokens = simp_tokenize(string)
        tokens = pos_tag(tokens)
        source = kt_tokenize(string)
        ktpos_alters = []
        for i, (word, tag) in enumerate(tokens):
            if source[i] and self.is_keyword_tag(tag):
                ktpos_alters.append(i)
        _, keywords = [], []
        for id in ktpos_alters:
            # if is_candiword(source[id]):
            keywords.append(source[id])
        return list(set(keywords))

    def politeness_marker_extract(self, string):
        df = pd.DataFrame({'text':[string]})
        timestamp = [None]*len(df)
        reply_to = [None]*len(df)
        speaker = [None]*len(df)
        conversation_id = [1]*len(df)
        id = [1]*len(df)
        df['timestamp'] = timestamp
        df['reply_to'] = reply_to
        df['conversation_id'] = conversation_id
        df['id'] = id
        df['speaker'] = speaker
        speaker_df = pd.DataFrame(df['speaker'])
        speaker_df.rename(columns={'speaker':'id'},inplace=True)
        convo_df = pd.DataFrame(df['id'])
        corpus = Corpus.from_pandas(utterances_df=df, speakers_df=speaker_df, conversations_df=convo_df)
        parser = TextParser(verbosity=0)
        corpus = parser.transform(corpus)
        ps = PolitenessStrategies(verbose=0)
        corpus = ps.transform(corpus, markers=True)
        polite_strategy_markers = []
        output = {}
        formatted_output = {}
        politeness_strategies = {
            'positive': ['Deference', 'Gratitude', 'Indirect_(greeting)', 'HASPOSITIVE', 'HASNEGATIVE'],
            'negative': ['Please', 'Please_start', 'Indirect_(btw)', 'HASHEDGE', 'Hedges', 'Factuality', 'Apologizing', '1st_person_pl', '1st_person', '1st_person_start', '2nd_person', '2nd_person_start', 'Direct_question', 'Direct_start', 'SUBJUNCTIVE', 'INDICATIVE']
        }
        formatted_politeness_dict = {key: {} for key in politeness_strategies}

        for utterance in corpus.iter_utterances():
            uid  = utterance.id
            utt = corpus.get_utterance(str(uid))
            for ((k,v),(k1,v2)) in zip(utt.meta["politeness_strategies"].items(),utt.meta["politeness_markers"].items()):
                if v != 0:
                    output[k[21:len(k)-2]] = v2
                    for key, value in output.items():
                        markers = []
                        for marker in value:
                            for item in marker: 
                                markers.append(item[0])
                        formatted_output[key] = markers

        for strategy_type, strategies in politeness_strategies.items():
            for strategy in strategies:
                if strategy in formatted_output:
                    formatted_politeness_dict[strategy_type][strategy] = formatted_output[strategy]
        return formatted_politeness_dict