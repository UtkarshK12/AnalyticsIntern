#members
#utkarsh kumar- utkarsh.iit.delhi@gmail.com
# Loading the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import nltk
nltk.download('stopwords')
# %matplotlib inline
from sklearn.pipeline import Pipeline

# Loading dataset
df = pd.read_excel('Scraped_Data.xlsx')

df.shape

# Checking for null values
df.isnull().sum()

# Removing rows with null values
df = df[df.category.isnull() == False]

df.shape

# Converting caption to string
df['caption'] = df['caption'].astype(str)

# Cleaning the text data
stemmer = PorterStemmer()
words = stopwords.words("english")
df['caption'] = df['caption'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z0-9]", " ", x).split() if i not in words]).lower())

vectorizer = TfidfVectorizer(min_df=3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))

# Extracting 1000 best features from the text data
pipe = Pipeline([('vect', vectorizer), ('chi',  SelectKBest(chi2, k=1000))])
a = pipe.fit_transform(df['caption'], df['category'])
text_features = a.toarray()

text_features.shape
df['category'] = pd.Categorical(df['category'])

# Creating training and test datasets
X = text_features
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fitting the model
model = LinearSVC(class_weight='balanced', max_iter=5000)

model.fit(X_train, y_train)

# Performing prediction on test dataset
preds = model.predict(X_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, preds))
print('Final prediction f1 score: [%.8f]' % f1_score(y_test, preds, average='weighted'))

# Printing classification report and confusion matrix
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

