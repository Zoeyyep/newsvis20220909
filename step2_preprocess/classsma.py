import csv
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
pd.options.mode.chained_assignment = None  # default='warn'

class SentAnalysis:
    @staticmethod
    def nltkSentiment(view):
        header = ['neg', 'neu', 'pos', 'compound']
        sid = SentimentIntensityAnalyzer()
        for article in view:
            print(article)
            ss = sid.polarity_scores(article)
            print(ss)
            with open("score29.csv", 'a', newline='', encoding='utf-8') as f:
                dict_writer = csv.DictWriter(f, fieldnames=header)
                dict_writer.writerow(ss)