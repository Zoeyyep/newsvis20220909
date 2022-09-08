# TEXT VISUALIZATION OF NEWS ARTICLES
Welcome to my GitHub! 

A video demonstration of the program may be found at: https://youtu.be/b3sXL92dE-U

Note: Download all fiels in UI_final, then change the corresponding file path in the main code Mainnew.py to your own file path!

<!-- 
  <<< Author notes: Step 1 >>> 
-->
<details id=1 open>
<summary><h2>Step 1: Data source </h2></summary>

The data used in this project is a subset of All the News 2.0. The original data includes 250 million news articles from over 27 U.S. publications, so we ranked publishers based on the number of articles and selected the top six publishers: Reuters, The New York Times, CNBC, The Hill, People, and CNNA database of 11,852 articles was created by randomly selecting 2,000 articles from each publisher, after removing empty articles. The data size was 40.9 MB. The filtered dataset was created with six features: Type, Id, NewsSource, Year, Title, and Article. The py files in step1_sample have details instrucions.

Cause the dataset was too big to upload, so I put the link the following link: https://pan.baidu.com/s/1TMhi9NTesPVDTjse_DXkTQ **password**: 9gp1 


**A larger version of this dataset is now available at Components**：https://components.one/datasets/all-the-news-2-news-articles-dataset/

<!-- 
  <<< Author notes: Step 2 >>>
-->

<details id=2 open>
<summary><h2>Step 2: Data prepocessing</h2></summary>
 
The main concept of data processing is to read text data from a csv file, calculate sentiment analysis scores, and store them in a csv file. A major challenge in the data processing stage is maintaining the readability of the text, since text data can be distorted during processing, making it unreadable or meaningless. As a result, word morphology reduction (Lemmatization) is an important part of text preprocessing. Instead of stemming, we chose lemmatization because stemming removes only the affixes of words and extracts the major part of the word, which is normally the word in the dictionary, but stemming is not necessarily apparent in the word. In this way, ambiguity and poor reading will be avoided. In this project, using nltk.pos_tag() to get the lexicality of the word in the sentence, combined with word form reduction, can improve the performance of completing word form reduction very well. This is followed by using TextBlob to check spelling and reduce errors. More details in step2_preprocess file.
  
  
 <!-- 
  <<< Author notes: Step 3 >>> 
-->
<details id=3 open>
<summary><h2>Step 3: Data Visualization</h2></summary>

The system includes four levels in the visualization of data text generation, firstly, the data itself, which conveys the basic information of news, such as source, time, and content, in the form of a table; secondly, the high-frequency information appearing in the news content is explored for sentiment analysis display, which is shown in the form of a tree diagram and a table. The third layer is a comprehensive display of sentiment analysis of news content, including bar stacked graphs and parallel coordinate graphs, and increases interactivity in the form of pop-up windows, which can facilitate readers' observation of potential trend changes in attitudes reported by news sources. Finally, the visualized data can be refined and enhanced by exploring word clouds for the high-frequency information mentioned in the second layer. More details in UI_final file.

  
 
