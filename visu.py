import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/hugodebes/onlineNews/main/OnlineNewsPopularity.csv")
    data.rename(columns=lambda x: x[1:] if len(x)>3 else x, inplace=True)
    return data
data = load_data()

def modif_data(data):
  data = data.loc[data["n_tokens_content"]!=0]
  del data["url"]
  return data

def main():
    page = st.sidebar.selectbox(
            "Summary",
            [
             "Global Presentation",                          
             "Overview & Variables' Creation",
             "Data Visualization",
            ],) 
    if page=="Global Presentation":
      global_info()
    elif page == "Overview & Variables' Creation":
      overview()
    elif page == "Data Visualization":     
      visu()

def global_info():
  st.title("Streamlit App: Predicting the popularity of Online News")
  st.header("Welcome to this project ! :wave: \n We will dive into the decisive factors that make an article goes viral.")
  st.markdown(
    """
    Nowadays, Blogging and writing articles is a major growing market in the internet. Thanks to easy-to-use tools and friendly tutorials,
    everyone can be an inspiring writer and develop and share their thoughts on any given subject.\n Unfortunately, most articles goes under the radar as the 
    readers are drown in too many choices. It is impossible to read the 4 million blog posts every day on the internet. To create and maintain an audience, every writer 
    elaborates some techniques based on their experiences and knowledge.\n Our goal, here, is to help them confirm or invalidate their thoughts with the use of Visualization,
    Machine Learning and Deep Learning.
    """)
  st.header("Our dataset :eyes:")
  st.markdown(
    """ To extract insights about news popularity, we need a dataset gathering a large amount of articles and infos related to them. We used the **_Online News Popularity_** dataset
    provided by the UCI Machine Learning Repository.This dataset is part of a research project aiming at predicting the popularity of an article on one hand and optimizing the features
    to improve it on the other hand. The interesting parts is we analyze article **prior** to their publication to help writers have a better understanding of their work before putting 
    it out to the world. The articles were published by Mashable (www.mashable.com). The authors determined that an article was successfull if it had more than 1400 shares.
    If you want to download it, follow this link : https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity """) 
  st.caption(
    """  
    K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
    Support System for Predicting the Popularity of Online News. Proceedings
    of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence,
    September, Coimbra, Portugal.
    """)

def visu():
  st.header("Data Visualization :moon:")
  dis_shares()
  st.markdown("""Now that we know better our dataset and its structure, we want to extract useful insights from our graphs. First, we made some research about how
  experts thought were the main criterias to determine the popularity of an article. The following charts are made to approve or disapprove their theories. Of course, 
  our dataset comes frome a unique source which is Mashable. It's probably not representative of the entire internet as we sould use other sources like Medium or Google News.
  We'll see if the principles apply for articles on Mashable for the period of 2013 to 2015.""")
  subjectivity_influence()
  polarity_influence()
  data_channel_influence()
  length_influence()
  length_subject()
  article_author()
  length_word_influence()
  visual_influence()
  day_influence()

def dis_shares():
  st.subheader("Distribution of shares in the dataset :bell: ")
  st.markdown("Our target variable is the number of shares per article. We want to know a little more about its distribution in our dataset.")
  plot  = st.radio("Actual or Log transformation",["Actual","Log"])
  fig,ax = plt.subplots(figsize=(17, 13))
  if plot == "Actual":
    url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/shares_articles.png'
    res = requests.get(url)
    img = Image.open(BytesIO(res.content))
    st.image(img)
  elif plot=="Log":
    url = "https://raw.githubusercontent.com/hugodebes/onlineNews/main/shares_articles_log.png"
    res = requests.get(url)
    img = Image.open(BytesIO(res.content))
    st.image(img)
  st.markdown("""This phenomena is not ruled by a normal distribution. Some Machine Learning models as the Logistic and Linear Regression or the Gaussian Naive Bayes have a gaussian assertion and might 
  not work in that way. To predict new results we could use other models who don't need the normality of their data like notably the desision trees. Or we could use a logarithmic 
  transformation to flatten the number of shares. You could do so by clicking on the button.""")
  st.markdown("""Also in the log distribution, we clearly understand why the authors of the research paper chose 1400 as the limit between a successfull and an unsuccessfull articles.
  Indeed, it is the median of the distribution""")
  st.markdown("""
  Those transformations will modify how we view our data and interpret it. For most of the visualization we will keep the original scale to better understand the meaning and 
  relationship between our variables. Unfortunately, we'll have to filter some of the outliers of our graphs. That's why in the first part, we'll discuss what are the main criteria 
  to make an article a success. On the second part, we'll focus on the super successful articles to see if the same rules abide.
  """)
  st.markdown(
      """ When we use the term "success", we refer to how the articles were considered as a success in the research paper which is equivalent to more than 1400 total shares.""")

def subjectivity_influence():
  st.subheader("Should we use our emotions or stick only to the facts ? :crystal_ball: ")
  st.markdown("Based on our research, we found that writers must face a dilemma between their role as author and journalist against themselves. They have to choose to let their emotions on a subject speak or rise above the fray.")
  st.markdown("""This process implies NLP techniques and precisely the notion of subjectivity. When facing a corpus, the subjectivity tries to quantify the emotional implication of 
  the author and their personal opinion against the facts. It's a float that lies in the range of [0,1] where 0 represents the maximum of facts and 1 the maximum opinion.
  """)
  st.markdown("Fortunately, we have a variable that contains this valuable information in our dataset global_subjectivity. ")
  st.markdown("The following graph compares the difference of numbers between successful and unsuccessful articles for a given subjectivity.")  
  url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/subjectivity.png'
  res = requests.get(url)
  img = Image.open(BytesIO(res.content))
  st.image(img)
  st.markdown("""As we can see above, we have a switch around 0.4 from negative values to a positive one. It simply means that there are more successful articles than unsuccessful
  ones when the subjectivity exceeds 0.4.""")
  st.markdown("To conclude, if our new writer wants to write the average successful article, he should let his emotions speak.")
  st.markdown("""This confirms the intuition that stories that evoke intense emotions tend to drive popularity. The content that triggers “high-arousal” emotions performs better online
   whereas content that sparks “low-arousal” emotions is less viral. This phenomenon could be a starting point to explain why fake news spread more rapidly since they are focused on
  emotions and sensationnal rather than objectivity and facts. """)

def polarity_influence():
  st.subheader("Should we transmit positive or negative news ? :yin_yang: ")
  st.markdown(""" 
      Now that we know that emotions play a role in the success of the article, we need to know which one we should use. Should our writer be a messenger for bad or good news ? """)
  st.markdown(""" 
      Once again, we can use NLP techniques and this time the polarity. The polarity quantifies the overall emotion in a corpus whether it is positive or negative. It is a float
       contained in the interval [-1,1] where -1 is the extreme negative and 1 the extreme positive.""")
  st.markdown(""" 
      The polarity has also been computed in the research paper in the global_sentiment_polarity column.The following graph represents the repartition between the successful and 
      unsuccessful articles. The more a category is represented the more it is high. """)
  temp_data=data
  temp_data["success"]=  [1 if i>1400 else 0 for i in temp_data["shares"]]
  fig = px.histogram(temp_data,x="global_sentiment_polarity",color="success",barmode="overlay",title="Distribution of articles classified by success depending on polarity ")
  st.plotly_chart(fig)
  st.markdown(""" 
      Like subjectivity there is a switch between the higher number between the successful articles and the unsuccessful ones. It seems like a better idea for our writer to use positive emotions.
      For the professors' Jonah Berger and Katherine L. Milkman : “positive content is more viral than negative content. """)

def length_influence():
  st.subheader("Brief News or Detailed Explanations ? :runner:")
  st.markdown(""" 
      Writing is a time-consuming activity but is it worth it ? Should you spend your time writing about every aspect of the subject or just give the main points to your subjects?""")
  st.markdown("""
  The answer seems in-between and most researchers agreed to an optimal length between 1000 and 3000 words. Most american people read about 250 words/min and the average time given to 
  an article is around 7 minutes. Of course, those numbers are overgeneralised and can vary from site to site.""")
  col1, col2 = st.columns(2)
  col1.metric("Percentage of successful articles within the optimal length","54.3%")
  col2.metric("Percentage of successful articles outside the optimal length","48.4%")
  st.markdown("In this table, it is clear that you statistically have more chance to produce a successful article (54%) if its length is between 1000 and 3000 words.")

def length_subject():
  st.subheader("The length is all about the subject :dizzy:")
  st.markdown("""Even with the table above, we can assume that the length will not be the same depending on the topic. Of course, some themes are more likely to increase
  the number of words to make the article cohesive and drive popularity.""")
  channel_dict = {'lifestyle':1,'entertainment':2,'bus':3,'socmed':4,'tech':5,'world':6}
  channel= data[["data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world"]]
  channel.rename(columns=lambda x: x[16:], inplace=True)
  channel= pd.DataFrame(channel.idxmax(axis=1))
  channel['shares'] = data['shares']
  channel = channel.loc[channel["shares"]<8000]
  channel["med"] = channel[0].map(dict(channel.groupby(0)["shares"].median()))
  channel["sort"] = channel[0].map(channel_dict)
  channel["word_count"] = data["n_tokens_content"]
  channel.sort_values("sort",inplace=True)
  channel.rename({0:"topic"},axis=1,inplace=True)
  fig = px.scatter(
    channel, y='shares', x='word_count', color='topic', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue',title= "Number of Share depending on the number of words per subject"
  )
  fig.update_layout(width=900,height=800)              
  st.plotly_chart(fig)
  st.markdown("""
  The first thing to notice is that we obtain very different linear regression between the themes. Unfortunately, the R square 
  is not significant enough since we only use one variable which is the number of words in the article. 
  """)
  st.markdown("""However, we notice some differences between the themes that are interesting. First of all, the coefficient is near 0 for social network, entertainment and world 
  related articles. It means that for those categories, the number of words does not matter. Then, the lifestyle, business and technology articles are more significant with a positive 
  coefficient. It tells us that writing longer articles in those categories is a good idea. Those topics can have a direct impact on our daily life, so it makes sense that we are
  willing to spend more time and more words on those.
  """)
  st.markdown("""Once our new writer will know which themes will be discussed in his text, he'll need to aim between 1000 and 3000 words. If it is business, tech or lifestyle related,
  approaching the upper limit could be a plus.
  """)

def article_author():
  return 1

def length_word_influence():
  st.subheader("What about vocabulary ? :tornado:")
  st.markdown("Now that our writer has a better idea of what to write, he just needs to do one thing : write. But should he use long elaborated words or should he stick to a conversational language ?")
  st.markdown("""The graph beside will help him to decide. It is a direct comparison between the number of shares per article and the average length of the words. Of course, the higher the average
  is, the higher the words are elaborated and digress from speaking language.
  """)  
  url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/avg_word.png'
  res = requests.get(url)
  img = Image.open(BytesIO(res.content))
  st.image(img)
  st.markdown("""As we can there's a negative regression between the two factors. It implies that the number of shares decreases as the average of letter per words increases. Unfortunately,
  the relation is not really significant but it can give us an overall tendency.
  """)
  st.markdown("""Our writer should write in a way that is understandable by everyone. It seems logical that an article readable by everyone will attract more audience than one filled with niche 
  vocabulary. Also, the message that our writer tries to deliver will be more easily heard.
  """)

def headlines_influence():
  st.subheader("Influence of the day")
  return 1

def visual_influence():
  st.subheader("Should you include Images ? :camera:")
  st.markdown("""Great ! Our article is done now, our writer wrote it overnight and is satisfied with his content. Now, he is wondering about including images but he is not convinced about their
  impact. Let's draw a difference graph between the successful and unsuccessful articles considering the number of images.
  """) 
  url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/img.png'
  res = requests.get(url)
  img = Image.open(BytesIO(res.content))
  st.image(img)
  st.markdown("""The more images are displayed the more chance you have to be popular as we see an impressive switch around 2 images.We live in a society filled with images and videos 
  all the time and we are now more accustumed to reading descriptions on Instagram than long text daily. Of course, the book industry is still well
  alive but on the subway, the "graphic" social media have replaced the novel for the most part. We are now used to seeing images alongside texts.
  """)
  st.markdown("""Also, images can be a great help to memorize information and concept written in the article.
  For all those reasons, our writer should include images in his article. The content of the image is also fundamental but we do not possess data on them. It could be a great follow-up
  to this work.
  """)

def data_channel_influence():
  st.subheader("Influence of the channel :circus_tent:")
  st.markdown("Of course, what interests you. But if your only goal as a writer is to become successful there is probably some thematics that are more viral in Mashable and their audience. ")
  st.markdown(
      """How the theme and channel impact for the popularity of an article. It is strongly related to the link between the author and the audience of a site like Mashable. For example,
   if an author loves to write about home design, it will not find much success in a teenage site since it's not their main interest. On the opposite a bussiness article will probably find an audiance
   in a site mostly followed by entrepreneurs.
      """)
  channel_dict = {'lifestyle':1,
               'entertainment':2,
               'bus':3,
               'socmed':4,
               'tech':5,
               'world':6
              }
  channel= data[["data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world"]]
  channel.rename(columns=lambda x: x[16:], inplace=True)
  channel= pd.DataFrame(channel.idxmax(axis=1))
  channel['shares'] = data['shares']
  channel = channel.loc[channel["shares"]<8000]
  channel["med"] = channel[0].map(dict(channel.groupby(0)["shares"].median()))
  channel["sort"] = channel[0].map(channel_dict)
  channel.sort_values("sort",inplace=True)
  channel.rename({0:"topic"},axis=1,inplace=True)
  fig = px.box(channel, y="shares", facet_col="topic", color="topic",
             boxmode="overlay", points='suspectedoutliers')   
  fig.update_layout(width=800,height=800)              
  st.plotly_chart(fig)
  st.markdown(
    """
    The channel is an important factors as we notice multiple gaps. The articles in the channel world and entertainment did not attract much enthusiasm and have the lowest median share.
    The lifestyle and technology channel follow up with good results and median above the symbolic limit of 1400. It means that more than 50% percent of those articles were cnsidered successful. Finally,
    the social media channel is the grand winner in term of popularity : the first quartile is equal to 1300 which means almost 75% of articles in this channel are successfull.
    """)

def day_influence():
  st.subheader("When should you publish ? :calendar:")
  st.markdown(
      """
      Finally, our article is done and our writer can't wait to let the world know his work. However, he is not sure when to publish it ?

      Does the day of the publication influence how well an article perform ? It can be clearly be a factor as most people could be too busy to read at certain period. Also, 
      the weekend can have an impact since we have more free time.
      """)
  week_dict= {  'monday':1,
               'tuesday':2,
               'wednesday':3,
               'thursday':4,
               'friday':5,
               'saturday':6,
               'sunday':7
              }
  weekday = data[["weekday_is_monday","weekday_is_tuesday","weekday_is_wednesday","weekday_is_thursday","weekday_is_friday","weekday_is_saturday","weekday_is_sunday"]]
  weekday.rename(columns=lambda x: x[11:], inplace=True)
  weekday = pd.DataFrame(weekday.idxmax(axis=1))
  weekday['shares'] = data['shares']
  weekday = weekday.loc[weekday["shares"]<8000]
  weekday["med"] = weekday[0].map(dict(weekday.groupby(0)["shares"].median()))
  weekday["sort"] = weekday[0].map(week_dict)
  weekday.rename({0:"day"},axis=1,inplace=True)
  weekday.sort_values("sort",inplace=True)
  fig = px.box(weekday, y="shares", facet_col="day", color="day",
             boxmode="overlay", points='suspectedoutliers')
  for i in range(0,len(week_dict)):
    fig.update_traces(name=list(week_dict.keys())[i], jitter=0, col=week_dict[list(week_dict.keys())[i]])
  fig.update_layout(width=800,height=800)
  st.plotly_chart(fig)
  st.markdown(
      """
      As we can see, there is a difference between weekdays and the weekend. The median is 1800 shares per article during the weekend and 1300 during the week.
      It means that the median article is far above the success median when it is published during the week-end !
      To conclude, it's an advantage to post during the weekend in Mashable. It is not true for all platforms as we have seen that it is better to post on Tuesday and Thursday on LinkedIn for example.
      """)

def overview():
  st.header("Overall Overview & Variables' Creation :earth_americas:")
  st.subheader("Get to know our data set :satellite_antenna:")
  st.markdown("The first thing to do is to make sure our data are clean.")
  st.dataframe(data.head())
  st.markdown(f"We have a total of {len(data)} columns and {len(data.columns)} variables")
  st.markdown("One of the important parts in preprocessing is to deal with incoherent values and null ones. Let's have a look for null values")
  st.metric("Number of Variable with at least 1 null value",0)
  st.markdown("Fortunately, there is no null values in the dataset which is an excellent news. No need to assign artificial values or delete rows.")
  st.markdown("To know better our data, we use the function __describe__ to create a simple statistical study for each variable.")
  st.dataframe(data.describe())
  st.markdown(
    """
    Each variable is numerical except the URL but we need to be careful because some of them are categorical like `is_weekend` or `data_channel_is_bus`.\n Oddly, 
    some articles have length of 0 as we can see in the min cell of `n_tokens_length`. Let's see how many articles are concerned by this incoherence.
    """)
  len_0 = data.loc[data["n_tokens_content"]==0]
  st.metric("Number of Articles with a Length=0",len(len_0))
  st.markdown(
    """It is an important number and we don't want to lose too many informations. Sometimes, one variable is wrongly created but there is useful information in 
    the other columns. We decide to look at the statistical results of those rows.
    """)
  st.dataframe(len_0.describe())
  st.markdown("Most of the variables have a mean of **0**. We can assume that we will not lose a huge part of information if we delete it.")
  st.subheader("Extraction of new Variables :flashlight:")
if __name__ == "__main__":
    main()
