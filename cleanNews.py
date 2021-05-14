# Katon Minhas
# Fake News Classification using ISOT dataset
# Import fake and real from ISOT dataset, export a single CSV file with extracted features

#%% Libraries

import pandas as pd
from textblob import TextBlob
import numpy as np
from matplotlib import pyplot as plt
import nltk
import statsmodels.api as sm
import profanity_check as prf
import scipy.stats as stats
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler

#%% Read in data(ISOT Dataset: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)


path = "C:/Users/Katon/Documents/JHU/CreatingAIEnabledSystems/ISOT_Dataset/"

fake = pd.read_csv(path+"Fake.csv")
real = pd.read_csv(path+"True.csv")


#%% Data cleaning

# drop unnecessary columns, make date first column, remove invalid rows
# drop columns, make date first column
fake.drop(fake.columns.difference(['title', 'text', 'date']), 1, inplace=True)
fake = fake[['date', 'title', 'text']]
fake['date'] = pd.to_datetime(fake['date'], errors='coerce', utc=True)
fake['title'].replace(' ', np.nan, inplace=True)
fake['text'].replace(' ', np.nan, inplace=True)
fake = fake.dropna()


real.drop(real.columns.difference(['title', 'text', 'date']), 1, inplace=True)
real = real[['date', 'title', 'text']]
real['date'] = pd.to_datetime(real['date'], errors='coerce', utc=True)
real['title'].replace(' ', np.nan, inplace=True)
real['text'].replace(' ', np.nan, inplace=True)
real = real.dropna()

#%% Get features

# Fake
fakeTitleSubj = np.empty((len(fake), 1))
fakeTextSubj = np.empty((len(fake), 1))

fakeTextPol = np.empty((len(fake), 1))
fakeTitlePol = np.empty((len(fake), 1))

fakeTitleWords = np.empty((len(fake)))
fakeTextWords = np.empty((len(fake), 1))

fakeTitleUnique = np.empty((len(fake), 1))
fakeTextUnique  = np.empty((len(fake), 1))

fakeTitleProf = np.empty((len(fake), 1))
fakeTextProf = np.empty((len(fake), 1))

# Real 
realTitleSubj = np.empty((len(real), 1))
realTextSubj = np.empty((len(real), 1))

realTitlePol = np.empty((len(real), 1))
realTextPol = np.empty((len(real), 1))

realTitleWords = np.empty((len(real), 1))
realTextWords = np.empty((len(real), 1))

realTextUnique  = np.empty((len(real), 1))
realTitleUnique = np.empty((len(real), 1))

realTitleProf = np.empty((len(real), 1))
realTextProf = np.empty((len(real), 1))


for i in range(len(fake)):
    fakeRow = fake.iloc[i,:]
    
    fakeTitle = fakeRow.title
    
    fakeText = fakeRow.text
    
    fakeTitleBlob = TextBlob(fakeTitle)
    fakeTextBlob = TextBlob(fakeText)
    
    fakeTitleSubj[i] = fakeTitleBlob.subjectivity
    fakeTitlePol[i] = fakeTitleBlob.polarity
    fakeTitleWords[i] = len(nltk.word_tokenize(fakeTitle))
    fakeTitleUnique[i] = len(np.unique(nltk.word_tokenize(fakeTitle)))
    fakeTitleProf[i] = prf.predict_prob([fakeTitle])
    
    fakeTextSubj[i] = fakeTextBlob.subjectivity
    fakeTextPol[i] = fakeTextBlob.polarity
    fakeTextWords[i] = len(nltk.word_tokenize(fakeText))
    fakeTextUnique[i] = len(np.unique(nltk.word_tokenize(fakeText)))
    fakeTextProf[i] = prf.predict_prob([fakeText])
    
    if i % 1000 == 0:
        print(i)
    
    #Get real news features
    if i < len(real):
        realRow = real.iloc[i,:]
        
        realTitle = realRow.title
        # remove first two words from real['text'] - which are usually CITYNAME, STATE (Reuters)
        realText = realRow.text.split(' ', 3)[3]
        
        realTitleBlob = TextBlob(realTitle)
        realTextBlob = TextBlob(realText)
        
        realTitleSubj[i] = realTitleBlob.subjectivity
        realTitlePol[i] = realTitleBlob.polarity
        realTitleWords[i] = len(nltk.word_tokenize(realTitle))
        realTitleUnique[i] = len(np.unique(nltk.word_tokenize(realTitle)))
        realTitleProf[i] = prf.predict_prob([realTitle])
        
        realTextSubj[i] = realTextBlob.subjectivity
        realTextPol[i] = realTextBlob.polarity
        realTextWords[i] = len(nltk.word_tokenize(realText))
        realTextUnique[i] = len(np.unique(nltk.word_tokenize(realText)))
        realTextProf[i] = prf.predict_prob([realText])


#%% add to dataframes
fake['title_subjectivity'] = fakeTitleSubj
fake['title_polarity'] = np.abs(fakeTitlePol)
fake['title_words'] = fakeTitleWords
fake['title_unique'] = fakeTitleUnique
fake['title_pct_unique'] = fake['title_unique'] / fake['title_words']
fake['title_profanity'] = fakeTitleProf

fake['text_subjectivity'] = fakeTextSubj
fake['text_polarity'] = np.abs(fakeTextPol)
fake['text_words'] = fakeTextWords
fake['text_unique'] = fakeTextUnique
fake['text_pct_unique'] = fake['text_unique'] / fake['text_words']
fake['text_profanity'] = fakeTextProf


real['title_subjectivity'] = realTitleSubj
real['title_polarity'] = np.abs(realTitlePol)
real['title_words'] = realTitleWords
real['title_unique'] = realTitleUnique
real['title_pct_unique'] = real['title_unique'] / real['title_words']
real['title_profanity'] = realTitleProf

real['text_subjectivity'] = realTextSubj
real['text_polarity'] = np.abs(realTextPol)
real['text_words'] = realTextWords
real['text_unique'] = realTextUnique
real['text_pct_unique'] = real['text_unique'] / real['text_words']
real['text_profanity'] = realTextProf


#%% Add fake/real label

fake['label'] = 0
real['label'] = 1


#%% create 1 dataframe

ISOT = pd.concat([fake, real])

#%% Keep relevant columns

ISOT_title = ISOT[['title_subjectivity', 'title_polarity', 'title_pct_unique', 'title_profanity', 'label']]
ISOT_text = ISOT[['text_subjectivity', 'text_polarity', 'text_pct_unique', 'text_profanity', 'label']]
ISOT_total = ISOT[['title_subjectivity', 'title_polarity', 'title_pct_unique', 'title_profanity', 
                   'text_subjectivity', 'text_polarity', 'text_pct_unique', 'text_profanity', 'label']]


#%% Remove na
ISOT_title = ISOT_title.dropna()
ISOT_text = ISOT_text.dropna()


#%% Normalize
scaler = MinMaxScaler()

ISOT_title = scaler.fit_transform(ISOT_title)
ISOT_text = scaler.fit_transform(ISOT_text)
ISOT_total = scaler.fit_transform(ISOT_total)


#%% Write to csv

np.savetxt(path+"Title.csv", ISOT_title, delimiter=",")
np.savetxt(path+"Text.csv", ISOT_text, delimiter=",")









