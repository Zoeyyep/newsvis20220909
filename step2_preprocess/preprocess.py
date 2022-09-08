from textblob import TextBlob
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import re
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
pd.options.mode.chained_assignment = None  # default='warn'

data= pd.read_csv("/Users/zhaoziyi/Desktop/sma/step1_sample/allsample.csv",encoding='utf8')
unused_words= pd.read_csv('/Users/zhaoziyi/Desktop/sma/step1_sample/stop_words.csv',encoding='utf8')
stop_words= stopwords.words('english')
for w in unused_words['Words']:
    stop_words.append(w)

contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "don’t": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y’all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "ain’t": "am not",
    "aren’t": "are not",
    "can’t": "cannot",
    "can’t’ve": "cannot have",
    "’cause": "because",
    "could’ve": "could have",
    "couldn’t": "could not",
    "couldn’t’ve": "could not have",
    "didn’t": "did not",
    "doesn’t": "does not",
    "don’t": "do not",
    "don’t": "do not",
    "hadn’t": "had not",
    "hadn’t’ve": "had not have",
    "hasn’t": "has not",
    "haven’t": "have not",
    "he’d": "he had",
    "he’d’ve": "he would have",
    "he’ll": "he will",
    "he’ll’ve": "he will have",
    "he’s": "he is",
    "how’d": "how did",
    "how’d’y": "how do you",
    "how’ll": "how will",
    "how’s": "how is",
    "i’d": "i would",
    "i’d’ve": "i would have",
    "i’ll": "i will",
    "i’ll’ve": "i will have",
    "i’m": "i am",
    "i’ve": "i have",
    "isn’t": "is not",
    "it’d": "it would",
    "it’d’ve": "it would have",
    "it’ll": "it will",
    "it’ll’ve": "it will have",
    "it’s": "it is",
    "let’s": "let us",
    "ma’am": "madam",
    "mayn’t": "may not",
    "might’ve": "might have",
    "mightn’t": "might not",
    "mightn’t’ve": "might not have",
    "must’ve": "must have",
    "mustn’t": "must not",
    "mustn’t’ve": "must not have",
    "needn’t": "need not",
    "needn’t’ve": "need not have",
    "o’clock": "of the clock",
    "oughtn’t": "ought not",
    "oughtn’t’ve": "ought not have",
    "shan’t": "shall not",
    "sha’n’t": "shall not",
    "shan’t’ve": "shall not have",
    "she’d": "she would",
    "she’d’ve": "she would have",
    "she’ll": "she will",
    "she’ll’ve": "she will have",
    "she’s": "she is",
    "should’ve": "should have",
    "shouldn’t": "should not",
    "shouldn’t’ve": "should not have",
    "so’ve": "so have",
    "so’s": "so is",
    "that’d": "that would",
    "that’d’ve": "that would have",
    "that’s": "that is",
    "there’d": "there would",
    "there’d’ve": "there would have",
    "there’s": "there is",
    "they’d": "they would",
    "they’d’ve": "they would have",
    "they’ll": "they will",
    "they’ll’ve": "they will have",
    "they’re": "they are",
    "they’ve": "they have",
    "to’ve": "to have",
    "wasn’t": "was not",
    "we’d": "we would",
    "we’d’ve": "we would have",
    "we’ll": "we will",
    "we’ll’ve": "we will have",
    "we’re": "we are",
    "we’ve": "we have",
    "weren’t": "were not",
    "what’ll": "what will",
    "what’ll’ve": "what will have",
    "what’re": "what are",
    "what’s": "what is",
    "what’ve": "what have",
    "when’s": "when is",
    "when’ve": "when have",
    "where’d": "where did",
    "where’s": "where is",
    "where’ve": "where have",
    "who’ll": "who will",
    "who’ll’ve": "who will have",
    "who’s": "who is",
    "who’ve": "who have",
    "why’s": "why is",
    "why’ve": "why have",
    "will’ve": "will have",
    "won’t": "will not",
    "won’t’ve": "will not have",
    "would’ve": "would have",
    "wouldn’t": "would not",
    "wouldn’t’ve": "would not have",
    "y’all": "you all",
    "y’all": "you all",
    "y’all’d": "you all would",
    "y’all’d’ve": "you all would have",
    "y’all’re": "you all are",
    "y’all’ve": "you all have",
    "you’d": "you would",
    "you’d’ve": "you would have",
    "you’ll": "you will",
    "you’ll’ve": "you will have",
    "you’re": "you are",
    "you’re": "you are",
    "you’ve": "you have",
}

"""
STEP 1  Article Processing
"""
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


# Function to clean the html from the article
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    rem_num = re.sub('[0-9]+', '', cleantext)
    return rem_num


# Function expand the contractions if there's any
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

#  Get the lexical properties of words
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# Function process the data
def text_processing(df):
    n = len(df['Article'])
    df['Article'] = df['Article'].apply(lambda x: cleanhtml(str(x)))
    df['Article'] = df['Article'].apply(lambda x: expand_contractions(str(x)))
    df['Article'] = df['Article'].apply(lambda x: word_tokenize(str(x)))
    df['Article'] = df['Article'].apply(lambda x: ' '.join([word for word in x if word not in (stop_words)]))

    for i in range(n):
        lemmatizer = WordNetLemmatizer()
        Text = df['Article'][i].split()
        tagged_sent = pos_tag(Text)  # Get the lexical properties of words
        words = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            words.append(lemmatizer.lemmatize(tag[0], pos=wordnet_pos))
        df['Article'][i] = words
        # print(df['Article'][i])

    # Removing the punctuation
    df['Article'] = df['Article'].apply(lambda x: ''.join(word for word in str(x) if word not in punctuation))

    # Removing the whitespace and double spaces
    df['Article'] = df['Article'].apply(lambda x: re.sub(' +', ' ', x))


    df['Article']= df['Article'].astype(str)
    df['Article'] = df['Article'].apply(lambda x: str(TextBlob(x).correct()))

    index = df[df['Article'] == 'nan'].index
    df.drop(index, inplace=True)
    df.to_csv('sample29.csv', encoding="utf-8")
    return df
data = text_processing(data)



"""
 Emotional Analysis
"""
import csv
index = data[data['Article'].isnull()].index
data.drop(index, inplace=True)
header = ['neg','neu','pos','compound']
def nltkSentiment(view):
    sid = SentimentIntensityAnalyzer()
    for article in view:
        print(article)
        ss = sid.polarity_scores(article)
        print(ss)
        with open("score29.csv",'a',newline='',encoding = 'utf-8') as f:
            dict_writer = csv.DictWriter(f,fieldnames=header)
            dict_writer.writerow(ss)
nltkSentiment(data['Article'])



"""
 TFIDF
"""

# A higher TFIDF value indicates that the feature word is more important for this text.
# max_df=0.5 : If a word occurs in more than 50% of the documents, it will be removed as it is considered non-discriminatory at corpus level
# Output the TF-IDF weight matrix for each sentence. Each sentence weight vector corresponds to the feature word list.

from sklearn.feature_extraction.text import TfidfVectorizer

WordList = list(data['Article'])
tfidf = TfidfVectorizer(use_idf=True, max_df=0.5, min_df=1)
vectors = tfidf.fit_transform(WordList)  # 生成文档的向量
# print(vectors)

# STEP 2 Constructs a dictionary (dict_of_tokens) where the keys are words and the values are TFIDF weights

dict_of_tokens = {i[1]: i[0] for i in tfidf.vocabulary_.items()}
tfidf_vectors = []  # all vectors by tfidf
for row in vectors:
    tfidf_vectors.append({dict_of_tokens[column]: value for (column, value) in zip(row.indices, row.data)})

# Take a look at the first document contained in this dictionary 看看这个字典包含的第一个文档
print("The number of document vectors=", len(tfidf_vectors),
      "\n The dictionary of document[0]:", tfidf_vectors[0])

# STEP 3 Sorting key phrases by TFIDF weights
doc_sorted_tfidfs = []
for dn in tfidf_vectors:
    newD = dict(sorted(dn.items(), key=lambda x: x[1], reverse=True))
    doc_sorted_tfidfs.append(newD)

# Get a list of keywords with no weight
tfidf_kw = []
for doc_tfidf in doc_sorted_tfidfs:
    ll = list(doc_tfidf.keys())
    tfidf_kw.append(ll)

# STEP 4 Extract the top 20 key phrases for each article, ranked by TFIDF weight

Top = 20
for i in range(len(tfidf_kw)):
    keywords = tfidf_kw[i][0:Top]
    # print(tfidf_kw[i][0:Top])
    with open("Keywords0809.csv", 'a', encoding='utf-8') as f:
        f.write(str(keywords) + '\n')

