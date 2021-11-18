# Analyzing Twitter Sentiment

Using Data Science Models to Read Sentiment in Tweets

Have you ever wondered how well a convention faired amongst the public? Well here in this notebook, we will go over real tweets that were sent out during and pertaining to the 2012 South by Southwest Festival. We will look into how people felt about the event as to gain insights on how the companies that took part can improve in the future.

### Methodology

The file used:

CrowdFlower via data.world ('judge-1377884607_tweet_product_company.csv')

### Filtering and Transformation

From this dataset there was a column that would not be used since it did not contain enough information as more than half the entries were empty. From there, other null values wre dropped and columns were renamed for better readability. We also focused solely on the positive and negative tweets since that would give a better indication to companies on what to improve, whether it be products or services. Next were the word frequencies charts to see if there were any indication of the dataset leaning one way or another. From that alone, the convention showed promise as there are more positive tweets but our models later on will be able tell us more.  

![frequenciescharts](https://user-images.githubusercontent.com/79724188/142332589-6c37ce1c-8b4c-459c-96ee-ff5123792401.png)

We then began transforming our data so that we can use them in our models more efficiently. First thing that was done was lemmatization. Lemmatizing the data will ensure that the words we are returning in our model are actual words as well as cutting down plural forms to singular so as to help reduce the repetition. Next is TF-IDF (or otherwise known as term frequency - inverse document frequency), which help us show how relevant a word is to the tweet. For instance, we have already done the frequencies of the words for both emotions, but TF-IDF will allow us to look at words that are more unique but still contribute significantly to the over all feel of the message. This will allow us to see which words may be more important and help fine tune which are problematic/exceptional.

After using the lemmatized data with TF-IDF, we can see with the sparse matrix in the notebook that there are 2,838 rows of text with 5,198 columns (or words in this case). In addition, we have the next output which tells us that the average word length of each tweet is only on average about nine to ten words long. There is also the fact that 99% of columns contain zero means that there are a lot of unique words throughout the tweets.

This could tell us that the tweets in this particular dataset are not very long, but are very distinctive from one another. We could infer that the people that are tweeting are just putting their first thoughts out and are not looking into event too closely. Or this could mean that people are not wanting to go into great detail about the event. It is hard to say for sure at this moment, but it should be mentioned that Twitter does have a limitation on how long a tweet can be so that should also be taken into account. We'll continue to transform our data so that we can get a more accurate picture.

We've also taken the measure of implementing Spacy onto our raw dataset so that we can compare it to our lemmatized and TF-IDF dataset. We will put both sets into our models below and this will show us if our cleaning of the code manually is the better/more efficient way or if perhaps the Spacy library is. What Spacy does for us is tokenizes our tweet text data as well as offers other additional features to understand large volumes of text. Next step, modeling!

# Models

### Random Forest Model - Part One

![RandomForestLemModel](https://user-images.githubusercontent.com/79724188/142333529-13e0e0ce-5283-4256-b951-d3b1c2a9626e.png)

The model we tried first was Random Forest since it has the feature importance attribute, which will be a big help in indicating exactly what companies need to pay attention to. Here with this model, we ran our manually edited dataset with it and the results weren't perfect, but not bad. We can see across accuracy, precision, recall and f-1 scores that this model had some hiccups, but wasn't too unreliable in predicting whether a tweet was negative or positive. We will keep this in mind when going through our next set of models.


### Random Forest Model - Part Two

![RandomForestSpacyModel](https://user-images.githubusercontent.com/79724188/142335071-367a6c1f-e7ff-462b-b5d1-a091610359b5.png)

This next Random Forest model was run with the Spacy dataset with less favorable results than the first. However, it's worth noting that this model has run with all of our data, including the neutral tweets that were not part of the manually sorted data created earlier. So if were are to use this model with this particular data, it is perhaps a better model for companies who are interested in the full spectrum of emotion, including apathy. 

### Gaussian Naive Bayes - Part One

![GNBLemModel](https://user-images.githubusercontent.com/79724188/142334080-54f94f7f-e5a5-4bb8-ae23-2812d0df6dc0.png)

Our next model is the Gaussian Naive Bayes, or gnb for short, which typically do well with classifications. Here with our specially assembled dataset, we can see that so far the Random Forest model is having better luck across the board. There are lower accuracy, precision, recall and f-1 scores in comparison but this is still a good model to try since gnb models are suited for multiple class predictions. 

### Gaussian Naive Bayes - Part Two

![GNBSpacyModel](https://user-images.githubusercontent.com/79724188/142334267-ed2b0b44-8d37-4f33-9fa4-34bf1727de7d.png)

Our final model is once again gnb but with our Spacy data. Like it's Random Forest Spacy counterpart, it also performed worse but again, did deal with neutral tweets in addition to positive and negative. The scores are still not that impressive however, which indicates that this is probably not the model for us. At least, in this time and scenario.

# Final Model and Results

![FrequenciesandImportances](https://user-images.githubusercontent.com/79724188/142334597-1b8e49a6-f10f-47dc-ab1c-02ed43085771.png)

After running a few models with different categorized datasets, the one model that could help companies in the future fine tune their products/systems to better appease the public is our original Random Forest model. In this scenario as well, since we really want to focus on the positive and negative tweets, this will mean we recommend not using Spacy as without it, the model is better at understanding the data. Especially when focusing on the f-1 scores, since they allow us to have a good balance between recall and precision.

The feature importances shown above will also allow companies to really hone in on which aspects of their companies/products that the public like and dislike. These graphs above indicate that for the 2012 South by Southwest Festival, iPhones were positively received from the public. In future conventions, Apple should feature more products like the iPhone. However, it was not all great for Apple since the company name did feature heavily in the negative tweets as well.

A recommendation for Apple would be to improve on the company's public image before the next event so that they are rated as highly as their products. In addition to Apple, Google should also enhance their internet presence/services or otherwise they may find themselves not part of the next convention.

