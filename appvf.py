import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import requests
from io import BytesIO

def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/hugodebes/onlineNews/main/OnlineNewsPopularity.csv")
    data.rename(columns=lambda x: x[1:] if len(x)>3 else x, inplace=True)
    data = data.loc[data["n_tokens_content"]!=0]
    return data
data = load_data()

def load_image(url):
    res = requests.get(url)
    img = Image.open(BytesIO(res.content))
    return img

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
             "Prediction Algorithm",
             "Follow up"
            ],) 
    if page=="Global Presentation":
      global_info()
    elif page == "Overview & Variables' Creation":
      overview()
    elif page == "Data Visualization":     
      visu()
    elif page == "Prediction Algorithm":
      predict()
    elif page == "Follow up":
      follow_up()

def global_info():
  st.title("Streamlit App: Predicting the popularity of Online News")
  st.header("Welcome to this project ! :wave: \n We will dive into the decisive factors that make an article goes viral.")
  st.markdown(
    """
    Nowadays, Blogging and writing articles is a major growing market on the internet. Thanks to easy-to-use tools and friendly tutorials,
    everyone can be an inspiring writer and develop and share their thoughts on any given subject. Unfortunately, most articles go under the radar as the 
    readers are drowned in too many choices. It is impossible to read the 4 million blog posts every day on the internet. To create and maintain an audience, every writer 
    elaborates some techniques based on their experiences and knowledge. Our goal, here, is to help them confirm or invalidate their thoughts with the use of Visualization,
    Machine Learning and Deep Learning.
    """)
  st.header("Our dataset :eyes:")
  st.markdown(
    """ To extract insights about news popularity, we need a dataset gathering a large number of articles and info related to them. We used the **_Online News Popularity_** dataset
    provided by the UCI Machine Learning Repository. This dataset is part of a research project aiming at predicting the popularity of an article on one hand and optimizing the features
    to improve it on the other hand. The interesting part is we analyzes article **prior** to their publication to help writers have a better understanding of their work before putting 
    it out to the world. The articles were published by Mashable (www.mashable.com). The authors determined that an article was successful if it had more than 1400 shares.
    If you want to download it, follow this link : https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity """) 
  st.caption(
    """  
    K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
    Support System for Predicting the Popularity of Online News. Proceedings
    of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence,
    September, Coimbra, Portugal.
    """)
  st.header("Acknowledgments")
  st.markdown("We would like to thank our lead teacher, Yann Kervella, for his advice and resources that allowed us to create this app.")

def visu():
  st.title("Data Visualization :moon:")
  dis_shares()
  st.markdown("""Now that we know better our dataset and its structure, we want to extract useful insights from our graphs. First, we made some research about how
  experts thought were the main criteria to determine the popularity of an article. The following charts are made to approve or disapprove their theories. Of course, 
  our dataset comes from a unique source which is Mashable. It's probably not representative of the entire internet as we should use other sources like Medium or Google News.
  We'll see if the principles apply for articles on Mashable for the period of 2013 to 2015.""")
  subjectivity_influence()
  polarity_influence()
  data_channel_influence()
  length_influence()
  length_subject()
  length_word_influence()
  visual_influence()
  day_influence()
  author_influence()

def dis_shares():
  st.header("Distribution of shares in the dataset :bell: ")
  st.markdown("Our target variable is the number of shares per article. We want to know a little more about its distribution in our dataset.")
  plot  = st.radio("Actual or Log transformation",["Actual","Log"])
  fig,ax = plt.subplots(figsize=(17, 13))
  if plot == "Actual":
    url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/shares_articles.png'
    img = load_image(url)
    st.image(img)
  elif plot=="Log":
    url = "https://raw.githubusercontent.com/hugodebes/onlineNews/main/shares_articles_log.png"
    img=load_image(url)
    st.image(img)
  st.markdown("""This phenomenon is not ruled by a normal distribution. Some Machine Learning models as the Logistic and Linear Regression or the Gaussian Naive Bayes have a gaussian assertion and might 
  not work in that way. To predict new results we could use other models that don't need the normality of their data like notably the decision trees. Or we could use a logarithmic 
  transformation to flatten the number of shares. You could do so by clicking on the button.""")
  st.markdown("""Also in the log distribution, we clearly understand why the authors of the research paper chose 1400 as the limit between successful and unsuccessful articles.
  Indeed, it is the median of the distribution""")
  st.markdown("""
  Those transformations will modify how we view our data and interpret it. For most of the visualization, we will keep the original scale to better understand the meaning and 
  relationship between our variables. Unfortunately, we'll have to filter some of the outliers of our graphs. That's why in the first part, we'll discuss what are the main criteria 
  to make an article a success. In the second part, we'll focus on the super successful articles to see if the same rules abide by.
  """)
  st.markdown(
      """When we use the term "success", we refer to the articles considered as a success in the research paper which is equivalent to more than 1400 total shares.""")

def subjectivity_influence():
  st.header("Should we use our emotions or stick only to the facts ? :crystal_ball: ")
  st.markdown("Based on our research, we found that writers must face a dilemma between their role as author and journalist against themselves. They have to choose to let their emotions on a subject speak or rise above the fray.")
  st.markdown("""This process implies NLP techniques and precisely the notion of subjectivity. When facing a corpus, the subjectivity tries to quantify the emotional implication of 
  the author and their personal opinion against the facts. It's a float that lies in the range of [0,1] where 0 represents the maximum of facts and 1 the maximum opinion.
  """)
  st.markdown("Fortunately, we have a variable that contains this valuable information in our dataset global_subjectivity. ")
  st.markdown("The following graph compares the difference of numbers between successful and unsuccessful articles for a given subjectivity.")
  url = "https://raw.githubusercontent.com/hugodebes/onlineNews/main/subjectivity.png"
  img= load_image(url)
  st.image(img)
  st.markdown("""As we can see above, we have a switch around 0.4 from negative values to a positive ones. It simply means that there are more successful articles than unsuccessful
  ones when the subjectivity exceeds 0.4.""")
  st.markdown("To conclude, if our new writer wants to write the average successful article, he should let his emotions speak.")
  st.markdown("""This confirms the intuition that stories that evoke intense emotions tend to drive popularity. The content that triggers “high-arousal” emotions performs better online
   whereas content that sparks “low-arousal” emotions is less viral. This phenomenon could be a starting point to explain why fake news spread more rapidly since they are focused on
  emotions and sensational rather than objectivity and facts.""")

def polarity_influence():
  st.header("Should we transmit positive or negative news ? :yin_yang: ")
  st.markdown(""" 
      Now that we know that emotions play a role in the success of the article, we need to know which one we should use. Should our writer be a messenger for bad or good news ? """)
  st.markdown(""" 
      Once again, we can use NLP techniques and this time the polarity. The polarity quantifies the overall emotion in a corpus whether it is positive or negative. It is a float
       contained in the interval [-1,1] where -1 is the extreme negative and 1 the extreme positive.""")
  st.markdown(""" 
      The polarity has also been computed in the research paper in the global_sentiment_polarity column. The following graph represents the repartition between the successful and 
      unsuccessful articles. The more a category is represented the more it is high.""")
  temp_data=data
  temp_data["success"]=  [1 if i>1400 else 0 for i in temp_data["shares"]]
  fig = px.histogram(temp_data,x="global_sentiment_polarity",color="success",barmode="overlay",title="Distribution of articles classified by success depending on polarity ")
  st.plotly_chart(fig)
  st.markdown(""" 
      Like subjectivity, there is a switch between the higher number between the successful articles and the unsuccessful ones. It seems like a better idea for our writer to use positive emotions.
      For the professors' Jonah Berger and Katherine L. Milkman: “positive content is more viral than negative content. """)

def length_influence():
  st.header("Brief News or Detailed Explanations ? :runner:")
  st.markdown(""" 
      Writing is a time-consuming activity but is it worth it ? Should you spend your time writing about every aspect of the subject or just give the main points to your subjects?""")
  st.markdown("""
  The answer seems in-between and most researchers agreed to an optimal length between 1000 and 3000 words. Most American people read about 250 words/min and the average time given to 
  an article is around 7 minutes. Of course, those numbers are overgeneralised and can vary from site to site.""")
  col1, col2 = st.columns(2)
  col1.metric("Percentage of successful articles within the optimal length","54.3%")
  col2.metric("Percentage of successful articles outside the optimal length","48.4%")
  st.markdown("We have computed that you statistically have more chance to produce a successful article (54%) if its length is between 1000 and 3000 words.")

def length_subject():
  st.header("The length is all about the subject :dizzy:")
  st.markdown("""Even with the metrics above, we can assume that the length will not be the same depending on the topic. Of course, some themes are more likely to increase
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
  The first thing to notice is that we obtain very different linear regressions between the themes. Unfortunately, the R square 
  is not significant enough since we only use one variable. 
  """)
  st.markdown("""However, we notice some differences between the themes that are interesting. First of all, the coefficient is near 0 for social network, entertainment and world 
  related articles. It means that for those categories, the number of words does not matter. Then, the lifestyle, business and technology articles are more significant with a positive 
  coefficient. It tells us that writing longer articles in those categories is a good idea. Those topics can have a direct impact on our daily life, so it makes sense that we are
  willing to spend more time and more words on those.
  """)
  st.markdown("""Once our new writer will know which themes will be discussed in his text, he'll need to aim between 1000 and 3000 words. If it is business, tech or lifestyle-related,
  approaching the upper limit could be a plus.
  """)


def length_word_influence():
  st.header("What about vocabulary ? :tornado:")
  st.markdown("Now that our writer has a better idea of what to write, he just needs to do one thing: write. But should he use long elaborated words or should he stick to a conversational language?")
  st.markdown("""The graph beside will help him to decide. It is a direct comparison between the number of shares per article and the average length of the words. Of course, the higher the average
  is, the higher the words are elaborated and digress from speaking language.
  """)
  res = Image.open('/content/drive/MyDrive/ESILV/Image_OnlineNews/avg_word.png')
  st.image(res)
  st.markdown("""As we can there's a negative regression between the two factors. It implies that the number of shares decreases as the average of letters per words increases. Unfortunately,
  the relation is not really significant but it can give us an overall tendency.
  """)
  st.markdown("""Our writer should write in a way that is understandable to everyone. It seems logical that an article readable by everyone will attract more audience than one filled with niche 
  vocabulary. Also, the message that our writer tries to deliver will be more easily heard.
  """)

def visual_influence():
  st.header("Should you include Images ? :camera:")
  st.markdown("""Great ! Our article is done now, our writer wrote it overnight and is satisfied with his content. Now, he is wondering about including images but he is not convinced about their
  impact. Let's draw a difference graph between the successful and unsuccessful articles considering the number of images.
  """)
  url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/avg_word.png'
  img=load_image(url)
  st.image(img)
  st.markdown("""The more images are displayed the more chance you have to be popular as we see an impressive switch around 2 images. We live in a society filled with images and videos 
  all the time and we are now more accustomed to reading descriptions on Instagram than long text daily. Of course, the book industry is still well
  alive but on the subway, the "graphic" social media have replaced the novel for the most part. We are now used to seeing images alongside texts.
  """)
  st.markdown("""Also, images can be a great help to memorize information and concept written in the article.
  For all those reasons, our writer should include images in his article. The content of the image is also fundamental but we do not possess data on them. It could be a great follow-up
  to this work.
  """)

def data_channel_influence():
  st.header("Influence of the channel :circus_tent:")
  st.markdown("Of course, what interests you. But if your only goal as a writer is to become successful there is probably some thematics that are more viral in Mashable and their audience. ")
  st.markdown(
      """How the theme and channel impact for the popularity of an article. It is strongly related to the link between the author and the audience of a site like Mashable. For example,
   if an author loves to write about home design, it will not find much success in a teenage site since it's not their main interest. On the opposite, a business article will probably find an audience
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
    The channel is an important factor as we notice multiple gaps. The articles in the channel world and entertainment did not attract much enthusiasm and have the lowest median share.
    The lifestyle and technology channel follow up with good results and median above the symbolic limit of 1400. It means that more than 50% per cent of those articles were considered successful. Finally,
    the social media channel is the grand winner in terms of popularity : the first quartile is equal to 1300 which means almost 75% of articles in this channel are successful.
    """)

def day_influence():
  st.header("When should you publish ? :calendar:")
  st.markdown(
      """
      Finally, our article is done and our writer can't wait to let the world know his work. However, he is not sure when to publish it ?

      Does the day of the publication influence how well an article perform ? It can clearly be a factor as most people could be too busy to read at a certain period. Also, 
      the weekend can have an impact since we have more free time.
      """)
  week_dict= { 'monday':1,
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

def author_influence():
  st.markdown("As explained earlier, playing the number games and post regularly is very efficient for most experts. Using the data gathered thanks to web Scrapping we wants to see if this is true.")
  data2 = pd.read_csv("https://raw.githubusercontent.com/hugodebes/onlineNews/main/OnlineNews_Authors.csv")
  mean_shares = data2.groupby(['author']).mean()[['same_author', 'shares']]
  mean_shares['author'] = mean_shares.index
  mean_shares.shares= np.exp(mean_shares.shares)
  mean_shares= mean_shares[mean_shares['same_author']>200]
  fig = px.scatter(mean_shares, x='same_author', y='shares', color='shares',
                 trendline="ols", hover_name="author", hover_data=["same_author", "shares"],
                 title='Mean shares depending on the number of articles written by a unique author')
  st.plotly_chart(fig)
  st.markdown("""As we can see, the trendline for the author had at least 200 articles doesn't confirm our theory. This can be partially explained by the fact that the people that post almost 2 articles a day during 
  2 years such as Sam Laird may not focus on the quality of their contents rather than quantity. Moreover, if they don't actively promote it on social media for example, they must not be really famous
  anyways.
  """)

def overview():
  st.title("Overall Overview & Variables' Creation :earth_americas:")
  st.header("Get to know our data set :satellite_antenna:")
  st.markdown("The first thing to do is to make sure our data are clean.")
  st.dataframe(data.head())
  st.markdown(f"We have a total of {len(data)} columns and {len(data.columns)} variables")
  st.markdown("One of the important parts of preprocessing is to deal with incoherent values and null ones. Let's have a look for null values")
  st.metric("Number of Variable with at least 1 null value",0)
  st.markdown("Fortunately, there is no null values in the dataset which is excellent news. No need to assign artificial values or delete rows.")
  st.markdown("To know better our data, we use the function __describe__ to create a simple statistical study for each variable.")
  st.dataframe(data.describe())
  st.markdown(
    """
    Each variable is numerical except the URL but we need to be careful because some of them are categorical like `is_weekend` or `data_channel_is_bus`. Oddly, 
    some articles have a length of 0 as we can see in the min cell of `n_tokens_length`. Let's see how many articles are concerned by this incoherence.
    """)
  len_0 = data.loc[data["n_tokens_content"]==0]
  st.metric("Number of Articles with a Length=0",len(len_0))
  st.markdown(
    """It is an important number and we don't want to lose too much information. Sometimes, one variable is wrongly created but there is useful information in 
    the other columns. We decide to look at the statistical results of those rows.
    """)
  st.dataframe(len_0.describe())
  st.markdown("Most of the variables have a mean of **0**. We can assume that we will not lose a huge part of information if we delete it.")
  st.header("Extraction of new Variables :flashlight:")
  st.markdown("""
  This dataset is obviously deep and contains already a lot of information. However, we figured out that some precious information was missing and decided to implement 
  new variables.
  """)
  st.markdown("""
  In the visualization part, we notice that the length of the articles was a major factor. Plus, thanks to our research, we created an interval regarding the optimal 
  length between 1000 to 3000 words. Finally, we added the binary column **optimal_length**. 0 represent a corpus outside the interval and 1 inside the optimal length.""")
  st.code("""data["optimal_length"] = [1 if (i >1000) & (i<3000) else 0 for i in data.n_tokens_content]""",language="python")
  st.header("Web Scrapping Authors 🕸️")
  st.markdown("""
  During our research, we found that playing the numbers game was very efficient. This means that the more articles you post, the more likely you are to have a popular article.
  Neetzan Zimmerman, who the Wall Street Journal called possibly “the most popular blogger working on the Web today”, shared in an interview with HubSpot.comthat, he posts 10 to
  15 times per day. Not every post went viral, but the larger the volume of stories, the greater the chances of one taking off.

  Thus, to study the impact of the number of articles written by an author of the shares. To do so, we decided to scrap the author's names directly from Mashable.com with BeautifulSoup,
  using SoupStrainer and cchardet library to fasten it. Reaching ~2 requests per second, it took almost 8h to get everything. From here, we created a new variable called same_author 
  compiling the number of articles done by the same author.

  Issues encountered:

    - Almost 2 sec per request at the beginning - Solved with SoupStrainer and cchardet library.
    - Spyder crashing or Mashable.com aborting connections due to too many requests - Solved by saving the author's list to a pickle file every 50 new.
  """)
  st.code("""
  def author_date(url):
  page = requests.get(url)
  strainer_author = SoupStrainer('a', attrs={"class": "underline-link"})
  soup = BeautifulSoup(page.content, 'lxml', parse_only=strainer_author)
  try:
    author = soup.get_text(strip=True)
  except:
    author='Unknown'
  return author
  """,language="python")
  st.markdown("Thanks to this code, we were able to know the author of an article and compute how many texts he had written over the two year.")

def predict():
  st.title("Prediction Algorithms 📰💫")
  st.header("Introduction & Preprocessing 💡♻️")
  st.markdown("""
  We do have a better understanding of how our data works and are correlated thanks to the preprocessing and Visualization section. The second goal of this project is to predict whether an
  article will be successful or not. We'll use multiple Machine Learning algorithms and a neural network in order to solve this problem. 
  """)
  st.subheader("Manage the outliers 👤💰")
  st.markdown("""
  However, some outliers still remain, especially while looking at the maximum value compared to the 3rd Quartile on specific predictors. Thus, to 
  continue the outliers treatment while preprocessing the data, we need to split our dataset to consider it as real brand-new values, not knowing any statistical information on it. After,
  we use LocalOutlierFactor library, allowing us to detect outliers by measuring the local deviation of the density of a given sample with respect to its neighbours 
  (100 neighbours - Minkowski distance).
  """)
  st.code("lof = LocalOutlierFactor(n_neighbors = 100)",language="python")
  st.markdown("""
  We have a different range of values for our variables so we need to scale them to give neutral weights to each one.
  Plus, we use a RobustScaler from sklearn.preprocessing library taking into account InterQuartile Range instead of MinMax Range to scale our variables that are not 
  already rates or binary predictors. It is also another method to reduce the impact of the outlier. Moreover, we have to select carefully each predictor. Indeed some are categorical
  variables already encoded (OneHotEncoder) or rates.
  """)
  st.code("rob_scale = preprocessing.RobustScaler()",language="python")
  st.markdown("""
  Our goal here is to predict whether an article is going to be popular or not prior to its publication. Then, it is important not to take into account any timeline information about
  the article. Indeed, the more an article stays online, the more likely it is to be shared multiple times!
  """)
  st.code("data_X = data.drop(['timedelta'], axis=1)",language="python")
  st.header("Machine Learning 🪄")
  st.subheader("Classification or Regression ? 📚🐍")
  st.markdown("""
  In the research paper, the authors wanted to predict whether an article was likely to be successful or not. In consequence, they set a threshold to 1400 over which the article was
  considered popular. However, we thought that 1400 was quite subjective, even based on the number of Mashable users at that time, and we aimed for a model as general as possible.

  Thus, our first objective was to predict the number of shares an article will receive after its publication nuanced by its timedelta (days passed since the article publication).

  But, after implementing some regression models, we faced an important MSE score and a R² close to 0. It cleary means that our models weren’t representing the data. The problem was 
  still there even after treating the outliers that we knew some models were sensible to and applying a logarithm to normalize the shares distribution’s curve (c.f Regression Exploration part).

  So we decided to return to the classification problem, with the 1400 threshold, and improve the previous scores by implementing recent models and performing GridSearch.
  """)
  st.subheader("Logistic Regression 🌊📐")
  intro_logreg ="""Our baseline model for this study will be the Logistic Regression. The logistic regression is a classification algorithm from sklearn.linear_model that computes the probability that
  an input X belongs to a class Y. This probability is found using the logistic function which only depends on two parameters. They are estimated via the Maximum Likelihood Estimation Method (MLE).
  Once we know the probability, we set a threshold to assign each input to a specific class.
  Logistic Regression has many advantages, such as:

    - Really easy to implement, interpret, and efficient to train.

    - Less inclined to over-fitting even in high dimensional datasets thanks to the Regularization (L1 and L2) techniques.

  However, we also have to consider that it requires average or no multicollinearity between independent variables, which is our case looking at the correlation heatmap. But it is tough to obtain 
  complex relationships using logistic regression. More powerful and compact algorithms such as Neural Networks can easily outperform this algorithm."""
  image_lien_logreg = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/log_reg.png'
  code_algo_logreg = "logreg = sk_lm.LogisticRegression(C= 0.8, max_iter= 500, solver= 'lbfgs').fit(X_train, y_train)"
  accuracy_logreg,precision_logreg,recall_logreg,f1_logreg,exec_logreg = 0.65,0.66,0.7,0.68,4.4
  model_template(intro_logreg,image_lien_logreg,code_algo_logreg,accuracy_logreg,precision_logreg,recall_logreg,f1_logreg,exec_logreg)

  st.subheader("Support Vector Classifier 🧇↗️")
  intro_svm ="""SVC (or **Support Vector Classifier**) is a machine learning algorithm belonging to the Support Vector Machine (SVM) family. SVM algorithms use the **vector component of the dataset's samples** 
  to determine a preferential orientation. More precisely, we plot each data item as a point in n-dimensional space (where n is the number of features you have) with the value of each feature being the value of a
  particular coordinate. Then, we perform classification by **finding the hyper-plane that differentiates** the two classes very well.
  SVC has many advantages, such as:

    - Effective in high dimensional spaces and very versatile thanks to the different kernels available.
    - Memory efficient, by using a subset of training points in the decision function (called support vectors)."""
  image_lien_svm = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/svm.png'
  code_algo_svm = "svc = SVC(class_weight= None, kernel= 'linear', shrinking= True).fit(X_train, y_train)"
  accuracy_svm,precision_svm,recall_svm,f1_svm,exec_svm = 0.65,0.67,0.69,0.68,295
  model_template(intro_svm,image_lien_svm,code_algo_svm,accuracy_svm,precision_svm,recall_svm,f1_svm,exec_svm)

  st.subheader("Random Forest Classifier 〽️🌳")
  intro_rf ="""For our third model, we use one of the algorithms that obtain a sizable score with brute force technique: Random Forest. The Random Forest algorithm is an ensemble method for classification or
  regression based on multiple learning decision trees trained on slightly different subsets of data.
  Random Forest has many advantages, such as:

    - An excellent trade-off between interpretability and efficiency. This allows, for example, to determine which features were decisive for obtaining a prediction while remaining accurate.

    - Reduce the risk of error and avoid overfitting by considering different subsets (low-correlated trees).

  However, one of the major drawbacks of Random Forest is that its computations may go far more complex compared to other algorithms. To deal with it, our strategy was to slowly narrow the range of 
  hyperparameters and search the number of trees afterwards.

  cf. crowd psychology: an intelligence stronger than human intelligence emerges from a group of people.
  """
  image_lien_rf ='https://raw.githubusercontent.com/hugodebes/onlineNews/main/rf.png'
  code_algo_rf = "rfg = sk_en.RandomForestClassifier(n_estimators= 400, max_depth= 40, max_features= 'sqrt',max_samples= 0.8, n_jobs=-1).fit(X_train, y_train)"
  accuracy_rf,precision_rf,recall_rf,f1_rf,exec_rf = 0.66,0.67,0.72,0.69,37.4
  model_template(intro_rf,image_lien_rf,code_algo_rf,accuracy_rf,precision_rf,recall_rf,f1_rf,exec_rf)

  st.subheader("Light Gradient Boosted Machine 📈⚙️")
  intro_lgbm ="""In this section, we use a fast and powerful gradient boosting framework also based on multiple decision trees learning: LGBM. Gradient boosting is a method of turning weak learners into strong ones. It relies heavily on the idea that the prediction of the next tree within a model will reduce prediction errors when mixed with previous models. LGMB is a recent model that is performing in Kaggle competitions, with even better results than NN or XGBoost algorithms. LGBM has many advantages, such as:

    - Trains faster and is more efficient using less memory.

    - Very efficient in general, with more complex trees via a leaf-wise growth approach (~DFS), even on large datasets.
  """
  image_lien_lgbm = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/lgbm.png'
  code_algo_lgbm = "lgbm=lgb.LGBMClassifier(n_estimators= 100, boosting_type= 'gbdt', colsample_bytree= 0.8, learning_rate= 0.08, max_depth= 20).fit(X_train, y_train)"
  accuracy_lgbm,precision_lgbm,recall_lgbm,f1_lgbm,exec_lgbm = 0.67,0.68,0.72,0.7,1.2
  model_template(intro_lgbm,image_lien_lgbm,code_algo_lgbm,accuracy_lgbm,precision_lgbm,recall_lgbm,f1_lgbm,exec_lgbm)

  st.header("Deep Learning 🧿☄️")
  st.markdown("""A neural network is an algorithmic model composed of several layers of computational units, called neurons. Each neuron receives as input the outputs of the previous neuron while continuously
  adjusting the weights given to the values thanks to an error back-propagation system. Here our neural network doesn't need to be too complex: an input layer leading to 3 hidden layers with the relu activation 
  function + a dropout layer and 2-neurons-layer output with sigmoid AF. Our compiler is made of Adam optimizer linked with a binary_crossentropy loss function and accuracy metric.

  Fully-connected neural networks, as we have here, has many advantages, such as:

    - Store information on the entire network, allowing to work with incomplete knowledge.
    - Have a distributed memory while having an important numerical strength that can perform more than one job at the same time and is fault-tolerant.

  However, it is crucial to take into account the drawbacks of such algorithms such as the hardware dependence (leading to high computational time) and especially the risk of overfitting. To deal with it,
  we split our training set into training and validation sets. Validation set that we use to measure the accuracy on an unknown set (val_accuracy).
  """)
  url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/nnn.png'
  img=load_image(url)
  st.image(img)
  st.subheader("Model Architecture :brain:")
  st.code( """model = tf.keras.Sequential([
                             tf.keras.layers.Dense(6, activation='relu', input_dim=X_train.shape[1]),
                             tf.keras.layers.Dense(100, activation='relu'),
                             tf.keras.layers.Dense(500, activation='relu'),
                             tf.keras.layers.Dropout(0.1),
                             tf.keras.layers.Dense(200, activation='relu'),
                             tf.keras.layers.Dense(50, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')
  ])""",language="python")
  st.subheader("Training 🪵")
  st.code("""model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  model.fit(X_train, y_train, epochs = 20, validation_data=(X_validation, y_validation))
  """,language="python")
  accuracy_dl,precision_dl,recall_dl,f1_dl,exec_dl = 0.65,0.66,0.69,0.67,144.1
  col3,col4,col5,col6,col7 = st.columns(5)
  col3.metric("Accuracy",str(accuracy_dl),str(round(taux(accuracy_dl,0.66),2))+"%")
  col4.metric("Precision",str(precision_dl),str(round(taux(precision_dl,0.66),2))+"%")
  col5.metric("Recall",str(recall_dl),str(round(taux(recall_dl,0.7),2))+"%")
  col6.metric("F1-Score",str(f1_dl),str(round(taux(f1_dl,0.68),2))+"%")
  col7.metric("Execution Time",str(exec_dl)+"s",str(round(taux(exec_dl,7.979),2))+"%")
  st.caption("""
  Every result is compared to the base model.
  """)
  st.header("Performance Comparisons 🪓🎲")

  comparison_dict = {'Accuracy':(0.67, 0.65, 0.66,0.65, 0.66),
                   'Precision':(0.68, 0.66, 0.67, 0.66, 0.67),
                   'Recall':(0.72, 0.7, 0.72, 0.69, 0.69),
                   'F1-Score':(0.7, 0.68, 0.69, 0.67, 0.68),
                   'Exec Time (sec)':(1.2, 4.3,37.4, 144.1, 295)}

  comparison_df = pd.DataFrame(comparison_dict) 
  comparison_df.index= ['LightGBM', 'Logistic Regression', 'Random Forest Class', 'Neural Network', 'SVC'] 
  st.dataframe(comparison_df)
  fig = go.Figure()
  colors = ['cornflowerblue', 'dodgerblue', 'steelblue', 'lightskyblue', 'royalblue']
  fig.add_trace(go.Bar(x=comparison_df.index, y=comparison_df['Accuracy'], marker_color=colors,
                     hovertext=comparison_df['Exec Time (sec)'], opacity=0.7, name='Accuracy - Exec Time'))

  fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Precision'],
                         mode='markers + lines', name='Precision', marker_color='darkred'))
  fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Recall'],
                         mode='markers + lines', name='Recall', marker_color='darkorange'))
  fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['F1-Score'],
                         mode='markers + lines', name='F1-Score', marker_color='gold'))
  fig.update_layout(title='Performance Comparison Graph')
  st.plotly_chart(fig)
  st.header("Regression Exploration with MLR 🔥🔍")
  st.markdown("""
  In the research paper, the authors wanted to predict whether an article was likely to be successful or not. In consequence, they set a threshold to 1400 over which the article was considered popular.
  However, we thought that 1400 was quite subjective, even based on the number of Mashable users at that time, and we aimed for a model as general as possible.

  Thus, our first objective was to predict the number of shares an article will receive after its publication nuanced by its timedelta(days passed since the article publication).

  To do so we consider the multiple linear regression model from sklearn.linear_model library LinearRegression. By definition, multiple linear regression is a mathematical regression method that extends simple 
  linear regression to describe target variations depending on several predictors variations.

  Linear Regression has the advantage of allowing a more accurate understanding of the relative influence of one or more predictors on the target variable as well as the relationship between the different 
  predictors themselves. However, linear regression is also very sensible to the collinearity of the data and outliers hence the importance of robust-scaling.
  """)
  url ="https://raw.githubusercontent.com/hugodebes/onlineNews/main/mlr.png"
  img=load_image(url)
  st.image(img)
  st.code("multi_lin_reg = sk_lm.LinearRegression().fit(X_train, y_train)",language="python")
  col3,col4,col5 = st.columns(3)
  col3.metric("MSE",str(0.79))
  col4.metric("R Squared",str(0.06))
  col5.metric("Execution Time",str(0.09)+"s")
  st.markdown("""
  As we can see, we have R² close to 0. It cleary means that our model isn't representing the data at all. The problem is still there even after treating the outliers that we know some models are sensitive to,
  and applying a logarithm to normalize the shares distribution’s curve.

  This can be explained by the fact that the timedelta is the only information about the temporality of the article and depending on the time passed since its publication, 2 articles with the same characteristics
  may have completely different share scores. The number of shares can be exponential and rise extremely fast from one day to another. This also brings the luck factor that can't be denied when you talk about 
  popularity. Indeed, you only need the right person at the right time to share your article somewhere, to see it going from 217 shares to 4000 overnight.

  Also, it is important to consider that the number of shares not only depends on the content and its author (promotion on social media, famous ?) but also on the platform. Mashable in 2013 is clearly not the
  same as Mashable in 2014. The website changes and its popularity too, as you can see below. Thus, you have to stand back from the raw dataset and adapt your predictions !
  """)
  url = 'https://raw.githubusercontent.com/hugodebes/onlineNews/main/mashable.png'
  img=load_image(url)
  st.image(img)

def model_template(intro,image_lien,code_algo,accuracy,precision,recall,f1,exec):
  st.markdown(intro)
  img=load_image(image_lien)
  st.image(img)
  st.code(code_algo,language="python")
  st.markdown("""
  To choose the hyperparameters, we perform a grid search and select the best results.
  """)
  col3,col4,col5,col6,col7 = st.columns(5)
  col3.metric("Accuracy",str(accuracy),str(round(taux(accuracy,0.66),2))+"%")
  col4.metric("Precision",str(precision),str(round(taux(precision,0.66),2))+"%")
  col5.metric("Recall",str(recall),str(round(taux(recall,0.7),2))+"%")
  col6.metric("F1-Score",str(f1),str(round(taux(f1,0.68),2))+"%")
  col7.metric("Execution Time",str(exec)+"s",str(round(taux(exec,7.979),2))+"%")
  st.caption("""
  Every result is compared to the base model.
  """)

def taux(prec,suiv):
  return ((prec-suiv)/prec)*100

def follow_up():
  st.title("What’s next ? 🚀")
  st.markdown("""
  We learnt valuable information about our dataset and manage to predict new results with satisfactory results. Now, we want to go deeper into our analysis and help writers during their writing process.
  """)
  st.header("Improvement of the data set 🧩")
  st.markdown("""
  We are aware that our data are centred only on a unique source which is Mashable. Plus, the period is really restraint at a time where the habits were not the same as today. For example, 
  in 2013, Apple had not released the IPhone 6, and Edward Snowden made its first revelations in June. So, some criteria may be irrelevant in 2022 and the number of shares was strongly related to the overall 
  health of Mashable. 

  To make it more actual and generalise our model to multiple audiences, the next task would be to retrieve data from other blogs, websites, newspapers, or social media. 
  """)
  st.header("Interactive and real-time prediction ⏱")
  st.markdown("""
  Our goal is to help writers who from most of them have no background in Machine Learning and NLP techniques. Knowing that we can not ask them to fill every variable of the dataset to get the result of the
  classification and it would be time-consuming. The development of the interactive and real-time prediction, we would need to implement every function of our dataset like the number of stop words or the polarity. 
  The user would only have to upload their text and then automatically get their prediction. 

  It would also be an interesting way to retrieve more data to continuously train the different models.

  To conclude, we loved to work on this project and hope to develop such solutions in the near future.

  Ugo DEMY & Hugo DEBES
  """)
if __name__ == "__main__":
    main()
