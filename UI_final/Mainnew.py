import os
import sys
import nltk
from nltk import collections
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from PyQt5.QtWebEngineWidgets import QWebEngineView
from wordcloud import WordCloud as WC
from UInew import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QUrl, Qt
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
import time
warnings.filterwarnings('ignore')



###————A LIST OF WIDGETS TO APPLY
shadow_elements = {
    "dashboard", "Label_1", "Label_2", "tableWidget","IMAGE" }

###————Main window class
class MainWindow (QMainWindow, Ui_MainWindow):
    '''
       The main class includes the following functions
    '''
    def __init__(self):
        '''
        Initialization function definition
        :return:
       '''
        super(MainWindow, self).__init__()
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        for x in shadow_elements:
            effect = QtWidgets.QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(18)
            effect.setXOffset(0)
            effect.setYOffset(0)
            effect.setColor(QColor(0,0,0,255))
            getattr(self.ui, x).setGraphicsEffect(effect)
        self.show()
        self.setMinimumSize(1100,660)
###————Welcome Page
        self.gif = QMovie('/Users/zhaoziyi/Desktop/sma/UI_final/images/tech.gif')
        self.ui.back_label.setScaledContents(True)
        self.ui.back_label.setMovie(self.gif)
        self.ui.back_label.setAlignment(Qt.AlignCenter)
        self.gif.start()
        self.ui.stackedWidget.setCurrentIndex(1)

###————Loading data
        self.orignal = pd.read_csv('/Users/zhaoziyi/Desktop/sma/UI_final/allsample.csv',encoding='utf8')#origanl data
        self.data = pd.read_csv('/Users/zhaoziyi/Desktop/sma/UI_final/final_vis.csv',encoding='utf8')#textvis data
        self.unsued_words = pd.read_csv('/Users/zhaoziyi/Desktop/sma/UI_final/stop_words.csv', encoding='utf8')
        self.stop_words = stopwords.words('english')
        for w in self.unsued_words['Words']:
            self.stop_words.append(w)
        self.data['Article'] = self.data['Article'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (self.stop_words)]))

###————Customization of GUI
        self.ui.comboBox_pub.setStyleSheet(" #comboBox_pub{\n""color: rgb(0, 0, 0);\n""}")
        self.ui.comboBox_year.setStyleSheet(" #comboBox_year{\n""color: rgb(0, 0, 0);\n" "}")
        self.ui.graphicsView.scene_img = QGraphicsScene()
        self.imgShow = QPixmap()

###————Adding signal and slot mechanisms
        self.ui.browser = QWebEngineView()
        self.ui.searchButton.clicked.connect(self.search_article)       # Button 0 Search Article

        self.ui.concordance.clicked.connect(self.concordancefunction)   # Button 1 Concordance Generation
        self.ui.pushButton.clicked.connect(self.treemap)
        self.ui.pushButton_2.clicked.connect(self.ngram)

        self.ui.starcked.clicked.connect(self.scorefunction)            # Button 2 Stacked Bar Chart
        self.ui.SUM.clicked.connect(self.sumstarcked)
        self.ui.MEAN.clicked.connect(self.meanstarcked)

        self.ui.parallel.clicked.connect(self.parallbrowser)            # Button 3 Parallel Visualization

        self.ui.word_cloud.clicked.connect(self.wordfunction)           # Button 4 Word Cloud
        self.ui.onegram.clicked.connect(self.unicloud)
        self.ui.bigram.clicked.connect(self.bicloud)
        self.ui.pushButton_3.clicked.connect(self.tricloud)

        self.ui.browser.page().profile().downloadRequested.connect(self.on_downloadRequested)  # download image


###————Function definition

    def search_article(self):
        '''
        Search articles based on criteria filtering
        :return:
        '''
        start = time.time()
        self.ui.stackedWidget.setCurrentIndex(5)
        self.ui.Article.show()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()
        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.data_new = self.orignal[['Id','NewsSource', 'Year', 'Title', 'Article']]
            else:
                self.data_new = self.orignal.loc[(self.orignal["NewsSource"] == self.Pub),['Id','NewsSource', 'Year', 'Title', 'Article']]
        else:
            if self.Pub == 'ALL':
                self.data_new = self.orignal.loc[(self.orignal["Year"] == int(self.year)),['Id','NewsSource', 'Year', 'Title', 'Article']]
            else:
                self.data_new = self.orignal.loc[(self.orignal["Year"] == int(self.year)) & (self.orignal["NewsSource"] == self.Pub),['Id','NewsSource', 'Year', 'Title', 'Article']]

        data_rows = self.data_new.shape[0]  # Get the number of rows in the table
        data_colunms = self.data_new.shape[1]  # Get the number of columns in the table
        data_header = self.data_new.columns.values.tolist()  # Get table header

        self.ui.tableWidget.setColumnCount(data_colunms)
        self.ui.tableWidget.setRowCount(data_rows)
        self.ui.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Set row list header
        self.ui.tableWidget.setHorizontalHeaderLabels(data_header)
        self.ui.tableWidget.horizontalHeader().resizeSection(0, 70)
        self.ui.tableWidget.horizontalHeader().resizeSection(2, 50)
        self.ui.tableWidget.horizontalHeader().resizeSection(4,350)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(300)
        self.ui.tableWidget.verticalHeader().setFixedWidth(50)
        self.ui.tableWidget.setStyleSheet("color: rgb(61,80,95);")

        for i in range(data_rows):  # Row Loop
            data_rows_values = self.data_new.iloc[[i]]  # Read in a row of data
            data_rows_values_array = np.array(data_rows_values)  # Put the row of data into an array
            data_rows_values_list = data_rows_values_array.tolist()[0]  # Convert this array to a list
            for j in range(data_colunms):  # column loop
                data_items_list = data_rows_values_list[j]  # Each element of the row list is placed in the column list
                data_items = str(data_items_list)  # The data is converted to a string
                newItem = QTableWidgetItem(data_items)  # The data of this string type is newly created as a tablewidget element
                newItem.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Display is left-centered and vertically centered
                #newItem.setBackground(QColor(61,80,95))
                self.ui.tableWidget.setItem(i, j, newItem)  # Display the newItem element in row i and column j of the table
        end = time.time()
        print("Search article unning time: %s seconds" % (end - start))

    def on_downloadRequested(self, download):
        '''
        Provide images for download
        :return:
        '''
        dialog = QtWidgets.QFileDialog()
        dialog.setDefaultSuffix(".png")
        path, _ = dialog.getSaveFileName(self, "Save File", os.path.join(os.getcwd(), "newplot.png"), "*.png")
        if path:
            download.setPath(path)
            download.accept()

    def concordancefunction(self):
        '''
           concordance function choose
           :return:
        '''
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.FREQUENCY.show()

    def ngram(self):
        '''
        generate n-gram phrase
        reference by:
        https://www.analyticsvidhya.com/blog/2021/11/nlp-tags-frequencies-unique-terms-n-grams/
        :return:
        '''
        start = time.time()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()
        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.condata = self.data
            else:
                self.condata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
        else:
            if self.Pub == 'ALL':
                self.condata = self.data.loc[(self.data["Year"] == int(self.year))]
            else:
                self.condata = self.data.loc[
                    (self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]

        self.condata['Article'] = [word for word in self.condata['Article'] if word not in self.stop_words]
        self.condata['Article'] = [s for s in self.condata['Article'] if len(s) != 0]

        # function to prepare n-grams
        def count_ngrams(lines, min_length=2, max_length=3):
            lengths = range(min_length, max_length + 1)
            ngrams = {length: collections.Counter() for length in lengths}
            queue = collections.deque(maxlen=max_length)

            # Helper function to add n-grams at start of current queue to dict
            def add_queue():
                current = tuple(queue)
                for length in lengths:
                    if len(current) >= length:
                        ngrams[length][current[:length]] += 1

            # Loop through all lines and words and add n-grams to dict
            for line in lines:
                for word in nltk.word_tokenize(line):
                    queue.append(word)
                    if len(queue) >= max_length:
                        add_queue()
            # Make sure we get the n-grams at the tail end of the queue
            while len(queue) > min_length:
                queue.popleft()
                add_queue()
            return ngrams

        # print(count_ngrams(Article))

        self.bigramtodf = pd.DataFrame({'bigrams': [], 'bigrams_freq': []})
        self.trigramtodf = pd.DataFrame({'trigrams': [], 'trigrams_freq': []})

        def ngram_freq(ngrams, num=100):
            for n in sorted(ngrams):
                #print('----{} most frequent {}-grams ----'.format(num, n))
                for gram, count in ngrams[n].most_common(num):
                    #print('{0}: {1}'.format(' '.join(gram), count))
                    if n == 2:
                        self.bigramtodf = self.bigramtodf.append({'bigrams': gram, 'bigrams_freq': count}, ignore_index=True)
                    else:
                        self.trigramtodf = self.trigramtodf.append({'trigrams': gram, 'trigrams_freq': count}, ignore_index=True)
                print('')

        ngram_freq(count_ngrams(self.condata['Article']))
        self.ngramdf = pd.concat([self.bigramtodf, self.trigramtodf], axis=1)
        self.ngramdf['bigrams'] = self.ngramdf['bigrams'].astype(str)
        self.ngramdf['bigrams'] = self.ngramdf['bigrams'].str.replace(r'[^\w\s]+', '')
        self.ngramdf['trigrams'] = self.ngramdf['trigrams'].astype(str)
        self.ngramdf['trigrams'] = self.ngramdf['trigrams'].str.replace(r'[^\w\s]+', '')
        self.ngramdf.to_csv('n-grams.csv', encoding="utf-8")

        self.fig1_2 = go.Figure(data=[go.Table(
            header=dict(values=list(self.ngramdf),
                        fill_color='paleturquoise',
                        align=['left', 'center']),
            cells=dict(values=[self.ngramdf.bigrams, self.ngramdf.bigrams_freq, self.ngramdf.trigrams, self.ngramdf.trigrams_freq],
                       fill_color='lavender',
                       align=['left', 'center'],
                       font_size=12,
                       height=30))])
        self.fig1_2.show()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.gridLayout.addWidget(self.ui.browser)
        self.ui.browser.setHtml(self.fig1_2.to_html(include_plotlyjs='cdn'))
        self.ui.FREQUENCY.show()
        end = time.time()
        print("N-gram running time: %s seconds" % (end - start))

    def treemap(self):
        '''
                concordance treemap(SA)
                :return:
        '''
        start = time.time()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()
        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.condata = self.data
            else:
                self.condata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
        else:
            if self.Pub == 'ALL':
                self.condata = self.data.loc[(self.data["Year"] == int(self.year))]
            else:
                self.condata = self.data.loc[(self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]

        wordlists = []
        for i in range(len(self.condata['Article'])):
            words = str(self.condata['Article'].iloc[i]).split()
            wordlists += words
        counts = {}
        for word in wordlists:
            counts[word] = counts.get(word, 0) + 1
        items = list(counts.items())
        items.sort(key=lambda x: x[1], reverse=True)
        frequency = pd.DataFrame(items)[0:1000]#   top 1000 words
        # print(frequency)
        sid = SentimentIntensityAnalyzer()
        ss = []
        for word in frequency[0]:
            score = sid.polarity_scores(word)
            ss.append(score)
        ss = pd.DataFrame(ss)
        result = pd.concat([frequency, ss], axis=1)
        result.rename(columns={0: 'Word', 1: 'Times', 'neg': 'Negative', 'neu': 'Neutral', 'pos': 'Positive',
                               'compound': 'Compound'}, inplace=True)
        for i in range(len(result)):
            if result.loc[i, "Negative"] == 1:
                result.loc[i, 'Classify'] = 'Negative'
            elif result.loc[i, "Neutral"] == 1:
                result.loc[i, 'Classify'] = 'Neutral'
            elif result.loc[i, "Positive"] == 1:
                result.loc[i, 'Classify'] = 'Positive'
            else:
                result.loc[i, 'Classify'] = 'No score'
        for i in range(len(result)):
            if result.loc[i, "Classify"] == 'Negative':
                result.loc[i, 'Value'] = -1
            elif result.loc[i, 'Classify'] == 'Neutral':
                result.loc[i, 'Value'] = 0
            elif result.loc[i, 'Classify'] == 'Positive':
                result.loc[i, 'Value'] = 1
            else:
                result.loc[i, 'Value'] = 2

        # stored concordance as a csv file
        result.to_csv('concordance.csv', encoding="utf-8")
        self.fig0 = px.treemap(result[0:2000],
                         path=['Classify', 'Word'],
                         values='Times',
                         color='Value'
                         )
        self.fig0.update_layout(
            coloraxis_colorbar=dict(
                title="Score Meaning",
                tickvals=[-1,0,1,2],
                ticktext=['-1 - Negative','0 - Neutral', '1 - Positive', '2 - No score']
            )
        )
        self.fig0.show()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.gridLayout.addWidget(self.ui.browser)
        self.ui.browser.setHtml(self.fig0.to_html(include_plotlyjs='cdn'))
        self.ui.FREQUENCY.show()
        end = time.time()
        print("Treemap running time: %s seconds" % (end - start))

    def scorefunction(self):
        '''
            starcked barchart visualisation
            :return:
        '''
        self.ui.stackedWidget.setCurrentIndex(2)
        self.ui.SMABARCHART.show()

    def sumstarcked(self):
        '''
             News Source SA score summation calculation
             :return:
        '''
        start = time.time()
        self.ui.stackedWidget.setCurrentIndex(2)
        self.ui.SMABARCHART.show()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()
        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.stardata = self.data
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].sum()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].sum()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].sum()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].sum()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                #load all data
                self.fig_all = go.Figure(data=
                                    px.bar(self.sumscore,
                                           x=['CNBC', 'CNN', 'People', 'Reuters', 'The Hill', 'The New York Times'],
                                           y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_all.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_all.to_html(include_plotlyjs='cdn'))

            else:
                self.stardata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].sum()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].sum()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].sum()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].sum()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                # Specified publisher, all years
                self.fig_pub = go.Figure(data=
                                    px.bar(self.sumscore, x=[self.Pub], y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_pub.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_pub.to_html(include_plotlyjs='cdn'))

        else:
            if self.Pub == 'ALL':
                self.stardata = self.data.loc[(self.data["Year"] == int(self.year))]
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].sum()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].sum()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].sum()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].sum()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                # Specified years , all publisher
                self.fig_year = go.Figure(data=
                                     px.bar(self.sumscore,
                                            x=['CNBC', 'CNN', 'People', 'Reuters', 'The Hill', 'The New York Times'],
                                            y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_year.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_year.to_html(include_plotlyjs='cdn'))

            else:
                self.stardata = self.data.loc[(self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].sum()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].sum()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].sum()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].sum()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                # Specified years , Specified publisher
                self.fig_ss = go.Figure(data=
                                   px.bar(self.sumscore, x=[self.Pub], y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_ss.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_ss.to_html(include_plotlyjs='cdn'))
        end = time.time()
        print("Sum starcked bar chart: %s seconds" % (end - start))


    def meanstarcked(self):
        '''
            News Source SA score mean calculation
           :return:
        '''
        start = time.time()
        self.ui.stackedWidget.setCurrentIndex(2)
        self.ui.SMABARCHART.show()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()

        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.stardata = self.data
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].mean()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].mean()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].mean()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].mean()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                self.fig_all = go.Figure(data=
                                         px.bar(self.sumscore,
                                                x=['CNBC', 'CNN', 'People', 'Reuters', 'The Hill',
                                                   'The New York Times'],
                                                y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_all.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_all.to_html(include_plotlyjs='cdn'))

            else:
                self.stardata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].mean()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].mean()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].mean()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].mean()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                self.fig_pub = go.Figure(data=
                                         px.bar(self.sumscore, x=[self.Pub],
                                                y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_pub.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_pub.to_html(include_plotlyjs='cdn'))

        else:
            if self.Pub == 'ALL':
                self.stardata = self.data.loc[(self.data["Year"] == int(self.year))]
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].mean()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].mean()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].mean()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].mean()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                self.fig_year = go.Figure(data=
                                          px.bar(self.sumscore,
                                                 x=['CNBC', 'CNN', 'People', 'Reuters', 'The Hill',
                                                    'The New York Times'],
                                                 y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_year.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_year.to_html(include_plotlyjs='cdn'))

            else:
                self.stardata = self.data.loc[
                    (self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]
                Compound = self.stardata.groupby(['NewsSource'])['Compound'].mean()
                Negative = self.stardata.groupby(['NewsSource'])['Negative'].mean()
                Neutral = self.stardata.groupby(['NewsSource'])['Neutral'].mean()
                Positive = self.stardata.groupby(['NewsSource'])['Positive'].mean()
                self.sumscore = pd.DataFrame([Negative, Neutral, Positive, Compound]).T
                self.fig_ss = go.Figure(data=
                                        px.bar(self.sumscore, x=[self.Pub],
                                               y=['Negative', 'Neutral', 'Positive', 'Compound']))
                self.fig_ss.show()
                self.ui.gridLayout_3.addWidget(self.ui.browser)
                self.ui.browser.setHtml(self.fig_ss.to_html(include_plotlyjs='cdn'))
        end = time.time()
        print("mean starcked bar chart: %s seconds" % (end - start))

    def parallbrowser(self):
        '''
        Parallel coordinate view, tested to filter real-time display
        :return:
        '''
        start = time.time()
        self.ui.stackedWidget.setCurrentIndex(3)
        self.ui.PARALELL.show()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()
        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.newdata = self.data
            else:
                self.newdata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
        else:
            if self.Pub == 'ALL':
                self.newdata = self.data.loc[(self.data["Year"] == int(self.year))]
            else:
                self.newdata = self.data.loc[(self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]

        self.fig2 = go.Figure(data=
        go.Parcoords(
            line=dict(color=self.newdata['Number'],
                      colorscale=[[0, 'purple'], [0.5, 'lightseagreen'], [1, 'gold']],
                      showscale=True),
            dimensions=list([
                dict(range = [1,6],
                    tickvals=[1, 2, 3, 4, 5, 6],
                     ticktext=['Reuters', 'The New York Times', 'CNBC', 'The Hill', 'People', 'CNN'],
                     label='NewsSource', values=self.newdata['Type']),
                dict(range=[0, 1],
                     label='Negative', values=self.newdata['Negative']),
                dict(range=[0, 1],
                     label='Positive', values=self.newdata['Positive']),
                dict(range=[0, 1],
                     visible=True,
                     label='Neutral', values=self.newdata['Neutral']),
                dict(range=[-1, 1],
                     label='Compound', values=self.newdata['Compound']),
                dict(range=[2016, 2020],
                     tickvals=[2016, 2017, 2018, 2019, 2020],
                     constraintrange=[1500, 2000],
                     label='Year', values=self.newdata['Year'])

            ])
        )
        )
        self.fig2.show()
        self.ui.gridLayout_4.addWidget(self.ui.browser)
        self.ui.browser.setHtml(self.fig2.to_html(include_plotlyjs='cdn'))
        end = time.time()
        print("Parallel coordinates running time: %s seconds" % (end - start))

    def wordfunction(self):
        '''
           word cloud display
           :return:
        '''
        self.ui.stackedWidget.setCurrentIndex(4)
        self.ui.WORDCLOUD.show()

    def unicloud(self):
        '''
            uni wordcloud
            :return:
        '''
        start = time.time()
        self.ui.stackedWidget.setCurrentIndex(4)
        self.ui.WORDCLOUD.show()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()
        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.clouddata = self.data
            else:
                self.clouddata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
        else:
            if self.Pub == 'ALL':
                self.clouddata = self.data.loc[(self.data["Year"] == int(self.year))]
            else:
                self.clouddata = self.data.loc[(self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]

        self.uni_cloud = WC(width=800,height=500,stopwords=self.stop_words,
            background_color='white').generate(str(self.clouddata['Article']))
        self.uni_cloud.to_file('uni_cloud.png')

        # Load the word cloud map into the UI
        self.imgShow.load('uni_cloud.png')
        self.imgShowItem = QGraphicsPixmapItem()
        self.imgShowItem.setPixmap(QPixmap(self.imgShow))
        self.ui.graphicsView.scene_img.addItem(self.imgShowItem)
        self.ui.graphicsView.setScene(self.ui.graphicsView.scene_img)
        self.ui.graphicsView.fitInView(QGraphicsPixmapItem(QPixmap(self.imgShow)))
        end = time.time()
        print("Unit word cloud Running time: %s seconds" % (end - start))


    def bicloud(self):
        '''
        binary word cloud
        :return:
        '''
        start = time.time()
        self.ui.stackedWidget.setCurrentIndex(4)
        self.ui.WORDCLOUD.show()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()

        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.bidata = self.data
            else:
                self.bidata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
        else:
            if self.Pub == 'ALL':
                self.bidata = self.data.loc[(self.data["Year"] == int(self.year))]
            else:
                self.bidata = self.data.loc[
                    (self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]

        self.bidata['Article'] = [word for word in self.bidata['Article'] if word not in self.stop_words]
        self.bidata['Article'] = [s for s in self.bidata['Article'] if len(s) != 0]

        # function to prepare n-grams
        def count_ngrams(lines, min_length=2, max_length=2):
            lengths = range(min_length, max_length + 1)
            ngrams = {length: collections.Counter() for length in lengths}
            queue = collections.deque(maxlen=max_length)
            # Helper function to add n-grams at start of current queue to dict
            def add_queue():
                current = tuple(queue)
                for length in lengths:
                    if len(current) >= length:
                        ngrams[length][current[:length]] += 1

            # Loop through all lines and words and add n-grams to dict
            for line in lines:
                for word in nltk.word_tokenize(line):
                    queue.append(word)
                    if len(queue) >= max_length:
                        add_queue()
            # Make sure we get the n-grams at the tail end of the queue
            while len(queue) > min_length:
                queue.popleft() ## 队首元素出队
                add_queue()
            return ngrams

        # print(count_ngrams(Article))

        self.bigramtodf = pd.DataFrame({'bigrams': [], 'bigrams_freq': []})

        def ngram_freq(ngrams, num=100):
            for n in sorted(ngrams):
                for gram, count in ngrams[n].most_common(num):
                    if n == 2:
                        self.bigramtodf = self.bigramtodf.append({'bigrams': gram, 'bigrams_freq': count},ignore_index=True)


        ngram_freq(count_ngrams(self.bidata['Article']))
        self.ngramdf = pd.concat([self.bigramtodf], axis=1)
        self.ngramdf['bigrams'] = self.ngramdf['bigrams'].astype(str)
        self.ngramdf['bigrams'] = self.ngramdf['bigrams'].str.replace(r'[^\w\s]+', '')
        self.ngramdf.to_csv('2-grams.csv', encoding="utf-8")

        self.bigramdic = dict(zip(self.ngramdf['bigrams'], self.ngramdf['bigrams_freq'].astype(int)))
        self.bicloud = WC(width=800,height=500,background_color='white').generate_from_frequencies(self.bigramdic)
        self.bicloud.to_file('bi_cloud.png')
        #generate(str(self.ngramdf['bigrams']))

        # Load the word cloud map into the UI
        self.imgShow.load('bi_cloud.png')
        self.imgShowItem = QGraphicsPixmapItem()
        self.imgShowItem.setPixmap(QPixmap(self.imgShow))
        self.ui.graphicsView.scene_img.addItem(self.imgShowItem)
        self.ui.graphicsView.setScene(self.ui.graphicsView.scene_img)
        self.ui.graphicsView.fitInView(QGraphicsPixmapItem(QPixmap(self.imgShow)))
        end = time.time()
        print("Binary word cloud running time: %s seconds" % (end - start))


    def tricloud(self):
        '''
        trinary cloud
        :return:
        '''
        start = time.time()
        self.ui.stackedWidget.setCurrentIndex(4)
        self.ui.WORDCLOUD.show()
        self.year = self.ui.comboBox_year.currentText()
        self.Pub = self.ui.comboBox_pub.currentText()

        if self.year == 'ALL':
            if self.Pub == 'ALL':
                self.tridata = self.data
            else:
                self.tridata = self.data.loc[(self.data["NewsSource"] == self.Pub)]
        else:
            if self.Pub == 'ALL':
                self.tridata = self.data.loc[(self.data["Year"] == int(self.year))]
            else:
                self.tridata = self.data.loc[
                    (self.data["Year"] == int(self.year)) & (self.data["NewsSource"] == self.Pub)]

        self.tridata['Article'] = [word for word in self.tridata['Article'] if word not in self.stop_words]
        self.tridata['Article'] = [s for s in self.tridata['Article'] if len(s) != 0]

        # function to prepare n-grams
        def count_ngrams(lines, min_length=2, max_length=3):
            lengths = range(min_length, max_length + 1)
            ngrams = {length: collections.Counter() for length in lengths}
            queue = collections.deque(maxlen=max_length)

            # Helper function to add n-grams at start of current queue to dict
            def add_queue():
                current = tuple(queue)
                for length in lengths:
                    if len(current) >= length:
                        ngrams[length][current[:length]] += 1

            # Loop through all lines and words and add n-grams to dict
            for line in lines:
                for word in nltk.word_tokenize(line):
                    queue.append(word)
                    if len(queue) >= max_length:
                        add_queue()
            # Make sure we get the n-grams at the tail end of the queue
            while len(queue) > min_length:
                queue.popleft()
                add_queue()
            return ngrams

        # print(count_ngrams(Article))
        self.trigramtodf = pd.DataFrame({'trigrams': [], 'trigrams_freq': []})

        def ngram_freq(ngrams, num=100):
            for n in sorted(ngrams):
                for gram, count in ngrams[n].most_common(num):
                    if n == 3:
                        self.trigramtodf = self.trigramtodf.append({'trigrams': gram, 'trigrams_freq': count},
                                                                   ignore_index=True)

        ngram_freq(count_ngrams(self.tridata['Article']))
        self.ngramdf = pd.concat([ self.trigramtodf], axis=1)
        self.ngramdf['trigrams'] = self.ngramdf['trigrams'].astype(str)
        self.ngramdf['trigrams'] = self.ngramdf['trigrams'].str.replace(r'[^\w\s]+', '')
        self.ngramdf.to_csv('3-grams.csv', encoding="utf-8")

        self.trigramdic = dict(zip(self.ngramdf['trigrams'], self.ngramdf['trigrams_freq'].astype(int)))
        self.triloud = WC(width=800, height=500,background_color='white').generate_from_frequencies(self.trigramdic)
        self.triloud.to_file('tri_cloud.png')

        # Load the word cloud map into the UI
        self.imgShow.load('tri_cloud.png')
        self.imgShowItem = QGraphicsPixmapItem()
        self.imgShowItem.setPixmap(QPixmap(self.imgShow))
        self.ui.graphicsView.scene_img.addItem(self.imgShowItem)
        self.ui.graphicsView.setScene(self.ui.graphicsView.scene_img)
        self.ui.graphicsView.fitInView(QGraphicsPixmapItem(QPixmap(self.imgShow)))
        end = time.time()
        print("Tinary word cloud running time: %s seconds" % (end - start))




## EXECUTE APP
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())







