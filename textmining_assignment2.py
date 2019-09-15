from selenium import webdriver
browser = webdriver.Safari() # opens the Safari browser
from bs4 import BeautifulSoup as bs
from selenium.common.exceptions import NoSuchElementException 
from selenium.common.exceptions import ElementNotVisibleException

page= "https://www.imdb.com/title/tt2631186/reviews?ref_=tt_ov_rt"
browser.get(page)
import time
reviews = []
i=1
# Below while loop is to load all the reviews into the browser till load more button dissapears
while (i>0):
    #i=i+25
    try:
        # Storing the load more button page xpath which we will be using it for click it through selenium 
        # for loading few more reviews
        button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]') # //*[@id="load-more-trigger"]
        button.click()
        time.sleep(5)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break
ps = browser.page_source 
soup=bs(ps,"html.parser")

reviews = soup.findAll("div",attrs={"class","text"})
for i in range(len(reviews)):
    reviews[i] = reviews[i].text

import pandas as pd
movie_reviews = pd.DataFrame(columns = ["reviews"])
movie_reviews["reviews"] = reviews

movie_reviews.to_csv("movie_reviews.csv",encoding="utf-8")
 

#here i am changing the directory to save on my required location
import os
os.getcwd()
os.chdir("/Users/darling/Desktop/DATA SCIENCE")

#refining the data
movie_reviews=[]
import re
movie_rev_string = "".join(movie_reviews) # Removing unwanted symbols incase if exists
movie_rev_string = re.sub("[^A-Za-z" "]+"," ",movie_rev_string).lower()
movie_rev_string = re.sub("[0-9" "]+"," ",movie_rev_string)   
#here we are splitting the words as individual string
movie_reviews_words = movie_rev_string.split(" ")

stop_words = stopwords.words('english')
with open("/Users/darling/Downloads/stop.txt","r") as sw:
    stopwords = sw.read()

movie_reviews_words = [w for w in movie_reviews_words if not w in stopwords]
movie_rev_string = " ".join(movie_reviews_words)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud_movie = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(movie_rev_string)

plt.imshow(wordcloud_movie)
#here we got the most used words with both postive words and negative words

with open("/Users/darling/Downloads/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")  
poswords = poswords[36:]

movie_pos_in_pos = (" ").join ([w for w in movie_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(movie_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
#here it shows the postive word cloud

with open("/Users/darling/Downloads/negative-words.txt",encoding = "ISO-8859-1") as neg:
  negwords = neg.read().split("\n")
negwords = negwords[37:]
movie_neg_in_neg = " ".join ([w for w in movie_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(movie_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)