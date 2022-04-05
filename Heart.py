#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# In[4]:


df = pd.read_csv("Heart_Failure.csv")

# In[5]:


df.head(6)

# In[6]:


df.shape

# In[7]:


df.isnull().sum()

# In[8]:


plt.figure(figsize=(20, 20))
ax = sns.boxplot(data=df)

# In[9]:


from scipy import stats

z = np.abs(stats.zscore(df))
print(z)

# In[10]:


threshold = 3
print(np.where(z > 3))  # The first array contains the list of row numbers and second array respective column numbers

# In[11]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# In[12]:


df = df[(z < 3).all(axis=1)]
df.shape

# In[13]:


df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape

# In[14]:


plt.figure(figsize=(20, 20))
ax = sns.boxplot(data=df)

# In[15]:


plt.figure(figsize=(20, 20))
d = sns.heatmap(df.corr(), cmap="coolwarm", annot=True)

# In[16]:


df.describe()

# In[17]:


from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca',
                                      'thal'])  # creating dummy variable
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']  # we have taken these columns for scale down
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# In[18]:


dataset.head()

# In[19]:


dataset.tail()

# In[20]:


dataset.describe()

# In[21]:


sns.pairplot(df, hue="target", height=3, aspect=1);

# In[22]:


y = dataset['target']
X = dataset.drop(['target'], axis=1)

# In[23]:


from sklearn.model_selection import train_test_split

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# In[25]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# In[26]:


knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
score = cross_val_score(knn_classifier, X_train, y_train, cv=10)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)

# In[27]:


score.mean()

# In[28]:


knn_classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                      metric_params=None, n_jobs=1, n_neighbors=5, p=1,
                                      weights='uniform')
knn_classifier.fit(X_train, y_train)
score = cross_val_score(knn_classifier, X_train, y_train, cv=10)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)

# In[29]:


score = cross_val_score(knn_classifier, X_train, y_train, cv=10)
score.mean()

# In[30]:


# confustion metrix
cm = confusion_matrix(y_test, y_pred_knn)
plt.title('Heatmap of Confusion Matrix', fontsize=15)
sns.heatmap(cm, annot=True)
plt.show()

# In[31]:


print(classification_report(y_test, y_pred_knn))

# In[32]:


import pickle

# save model
pickle.dump(knn_classifier, open('model.pkl', 'wb'))

# load model
Heart_disease_detector_model = pickle.load(open('model.pkl', 'rb'))

# predict the output
y_pred = Heart_disease_detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of K – Nearest Neighbor model: \n', confusion_matrix(y_test, y_pred), '\n')

# show the accuracy
print('Accuracy of K – Nearest Neighbor  model = ', accuracy_score(y_test, y_pred))

# In[ ]:


# In[ ]:
