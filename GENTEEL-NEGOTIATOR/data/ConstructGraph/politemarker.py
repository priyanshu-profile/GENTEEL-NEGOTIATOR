import pandas as pd
from convokit import PolitenessStrategies, Corpus, download, TextParser

def politeness_marker_extract(string):
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
        parser = TextParser(verbosity=1000)
        corpus = parser.transform(corpus)
        ps = PolitenessStrategies(verbose=100)
        corpus = ps.transform(corpus, markers=True)
        polite_strategy_markers = []
        output = {}
        formatted_output = {}
        politeness_strategies = {
            'positive': ['Deference', 'Gratitude', 'Indirect_(greeting)', 'HASPOSITIVE', 'HASNEGATIVE'],
            'negative': ['Please', 'Please_start', 'Indirect_(btw)', 'HASHEDGE', 'Hedges', 'Factuality', 'Apologizing', '1st_person_pl', '1st_person', '1st_person_start', '2nd_person', '2nd_person_start', 'Direct_question', 'Direct_start', 'SUBJUNCTIVE', 'INDICATIVE']
        }
        formatted_dict = {key: {} for key in politeness_strategies}

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

        
        print(output)
        print(formatted_output)

        for strategy_type, strategies in politeness_strategies.items():
            for strategy in strategies:
                if strategy in formatted_output:
                    formatted_dict[strategy_type][strategy] = formatted_output[strategy]
        print(formatted_dict)


#politeness_marker_extract("Hi JaGa. I am always puzzled as to why you ask me to do work which you are as capable of doing yourself as I am. We have spoken about this before. Wikipedia is a volunteer charity project, and people do their best in the time they have available to them, and people will tend to work initially in areas that interest them, and then help out on tedious tasks if they have the time or inclination - but nobody is compelled to do anything (well, other than to take care they are not doing harm). That particular splitting you are talking about was a long and complex one that nobody had done for more than two years because of the amount of work involved. Sending people a nag message at the end of it instead of pitching in and helping out yourself is not conducive to the spirit of support, co-operation and collaboration that embodies the spirit of Wikipedia that I respect and enjoy so much. If you spot a spelling mistake - fix it yourself instead of sending someone a message. If you see that an article needs sourcing, it's acceptable to put a general message on the article asking people who are interested in that sort of work to alert them, but it's even better to do the work yourself; it's not really done to pick on the last person who edited the article to ask them to do all the work. You may not have noticed but I did send a message to those people involved in that article letting them know what had happened, and that clean up work might now be needed as I am not an expert on the topic. I know you are well intentioned, but I have already indicated to you that I am uncomfortable with these messages. I would respect you much more if you pitched in and did the work yourself rather than send people these messages. I would love to know that you were helping out by tidying up after me. I would think that was great. Really I would. How about creating a template to be placed on newly created disamb pages that says that work on sorting them out needs to be done. And the template could put such articles into a category to enable editors to work through all the articles that need attention. I think a general message would be more in the spirit of Wikipedia than putting the weight all on one person. If we make a task too onerous for one person, then that task will be ignored.  <span style=\"border: 1px  #F10; background-color:cream;\">")
politeness_marker_extract("I like the refrigerator you're selling, but the price is a bit higher than what I was expecting. Can you offer any discounts?")
print()
politeness_marker_extract("I understand your concern, and I appreciate your interest. I'll do my best to accommodate your budget. How about a 5 percent discount, bringing the price down to $900? Please let me know if this works for you.")