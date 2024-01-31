import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


from wordcloud import WordCloud
import os


#Preprocessing Data
import nltk

#downloading
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import string, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer