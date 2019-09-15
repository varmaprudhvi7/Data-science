import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
macbook_reviews=[]

for i in range(1,30):
  mac=[]  
  url="https://www.flipkart.com/apple-macbook-air-core-i5-5th-gen-8-gb-128-gb-ssd-mac-os-sierra-mqd32hn-a-a1466/product-reviews/itmevcpqqhf6azn3?pid=COMEVCPQBXBDFJ8C&page="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("div",attrs={"class","qwjRop"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    mac.append(reviews[i].text)  
  macbook_reviews=macbook_reviews+mac 
#here we saving the extracted data 
with open("macbook.txt","w",encoding='utf8') as output:
    output.write(str(macbook_reviews))
    
# here we are refing the data 
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
os.getcwd()
os.chdir("/Users/darling/Desktop/DATA SCIENCE")  

# Joinining all the reviews into single paragraph 
mac_rev_string = " ".join(macbook_reviews) 

# Removing unwanted symbols incase if exists
mac_rev_string = re.sub("[^A-Za-z" "]+"," ",mac_rev_string).lower()
mac_rev_string = re.sub("[0-9" "]+"," ",mac_rev_string)   

#here we are splitting the words as individual string
mac_reviews_words = mac_rev_string.split(" ")
#removing the stop words
stop_words = stopwords.words('english')
with open("/Users/darling/Downloads/stop.txt","r") as sw:
    stopwords = sw.read()
temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]
mac_reviews_words = [w for w in mac_reviews_words if not w in stopwords]
mac_rev_string = " ".join(mac_reviews_words)
#creting word cloud for all words
wordcloud_mac = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(mac_rev_string)

plt.imshow(wordcloud_mac)
#here we are getting all words including postive words and negative words 
#now we are sepating postive words and negative words in word cloud
# positive words # Choose the path for +ve words stored in system
with open("/Users/darling/Downloads/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")  
poswords = poswords[36:]

mac_pos_in_pos = (" ").join ([w for w in mac_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(mac_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
#here we get wordcloud of allpostive words in reviews

# negative words  Choose path for -ve words stored in system
with open("/Users/darling/Downloads/negative-words.txt",encoding = "ISO-8859-1") as neg:
negwords = neg.split("\n")
negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
mac_neg_in_neg = " ".join ([w for w in mac_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(mac_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)
#here we are getting the most repeated negative Wordcloud