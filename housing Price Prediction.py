#!/usr/bin/env python
# coding: utf-8

# # Discription

# """This project involves predicting housing prices using a limited dataset with a unique challenge: strong multicollinearity among the predictor variables. The dataset includes various factors such as house area, number of bedrooms, furnished status, proximity to the main road, and potentially other relevant features. 
# The goal is to build a robust predictive model despite the intricate relationships among these variables."""

# # Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings;
warnings.filterwarnings('ignore')


# # Loading Dataset

# In[2]:


df=pd.read_csv(r"C:\Users\mandar joshi\OneDrive\Desktop\ds\ml\data sets\Housing.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# """There is no null value in that dataset so that we can directly go to the EDA part """

# # Data Visualization

# In[6]:


x=df.groupby(by='furnishingstatus').agg(price=('price','mean'),
                                       area=('area','mean'))
print(x)


# " Here'Groupby' gives us average value in  'price' and 'area' column according to each furniturestatus."

# In[7]:


plt.pie(labels=x.index,x=x['price'].values,autopct='%0.2f')


# "Here we can see that the price is distributed according to the furniture status, Higher to lower."

# In[8]:


plt.pie(labels=x.index,x=x['area'].values,autopct='%0.2f')


# In[9]:


plt.figure(figsize=(7,7))
sns.swarmplot(x=df['mainroad'],y=df['price'],hue=df['furnishingstatus'])


# "Here we can see that very less number of houses is there. They Don't have main road and mostly they are semi furnished."

# In[10]:


plt.figure(figsize=(7,7))
sns.swarmplot(x=df['mainroad'],y=df['area'],hue=df['furnishingstatus'])


# "Here we see the same thing."

# In[11]:


x=list(df.select_dtypes(include='object'))
print(x)
print(len(x))


# In[12]:


"These all are the categorical columns."


# In[13]:


plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=0.5,wspace=0.5)

for i in range(len(x)):
    plt.subplot(3,4,i+1)
    plt.xticks(rotation=90)
    sns.countplot(df[x[i]])


# "By applying the loop, here we plot all the categorical features in bar plot."

# In[14]:


sns.pairplot(df)
plt.show


# # Checking null values

# In[15]:


df.isna().sum()


# # Outlier Treatment

# "Here we are checking the outliers in our numerical columns with the help of box plot."

# In[16]:


df.boxplot('price')


# In[17]:


df.boxplot('area')


# In[18]:


df.head()


# In[19]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.histplot(df['price'], ax=axes[0])
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram of Price')

sns.histplot(df['area'], ax=axes[1])
axes[1].set_xlabel('Area')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Histogram of Area')


# """Here with the help of histogram we can see that our data is not normally distributed. So that  we use 'IQR method' for treating the outliers. 
# There is another method for 'Z score method'. But this method is used from when our data is normally distributed."""

# In[20]:


q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
IQR = q3 - q1
print(IQR)


# In[21]:


q2_1 = df['area'].quantile(0.25)
q2_3 = df['area'].quantile(0.75)
IQR2 = q2_3 - q2_1 
print(IQR2)


# In[22]:


#price
higher_limit_allowed = q3 + 1.5 * IQR
#area
higher_limit_allowed2 = q2_3 + 1.5 * IQR2


# In[23]:


higher_limit_allowed # for price


# In[24]:


higher_limit_allowed2 #for area


# In[25]:


df[df['price'] > higher_limit_allowed]


# In[26]:


df[df['area'] > higher_limit_allowed2]


# In[27]:


cleaned_df = df.copy()


#  "Here we are applying the capping method for treating the outlier in price and area column by setting the threshold value."

# In[28]:


cleaned_df['price'] = np.where(cleaned_df['price'] > higher_limit_allowed, higher_limit_allowed, cleaned_df['price'])


# In[29]:


cleaned_df['area'] = np.where(cleaned_df['area'] > higher_limit_allowed2, higher_limit_allowed2, cleaned_df['area'])


# In[30]:


cleaned_df.describe()


# In[31]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(cleaned_df['price'], ax=axes[0])
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram of Price')

sns.histplot(cleaned_df['area'], ax=axes[1])
axes[1].set_xlabel('Area')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Histogram of Area')


# In[32]:


"Now we can see that our data is normally distributed."


# # Correlation Analysis

# In[33]:


sns.heatmap(cleaned_df.corr(),annot=True)


# In[34]:


cleaned_df.columns


# In[35]:


cols=[i for i in cleaned_df.columns if cleaned_df[i].dtypes==object]
cols.pop(-1)
cols


# # Feature Engineering

# In[36]:


cleaned_df = pd.get_dummies(cleaned_df,columns=cols)
cleaned_df.head(10)


# In[37]:


cleaned_df['furnishingstatus']=cleaned_df['furnishingstatus'].replace({'furnished':0,'semi-furnished':1,'unfurnished':2})


# In[38]:


x=cleaned_df.drop('price',axis=1)
y=cleaned_df['price'] 


# In[39]:


x.head(2)


# In[40]:


y.head(2)


# # Splitting the data

# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)


# # Feature Scaling 

# In[42]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# """Here we are scaling our data with the use  with the standard scalar. There is Another approach is the 'min Max scaler'
# whenever data is Not normally distributed then We use the min Max scaler. Now our data is normally distributed, thats why we be used standard scaler."""

# # Model Validation

# """Here we trying to use the multiple Algorithms we have to minimise the error and got a better R2 square,
# That is the our first aim. """

# In[43]:


from sklearn.decomposition import PCA
pca = PCA(n_components=6)
x_train_pc = pca.fit_transform(x_train)
x_test_pc = pca.transform(x_test)


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error


# In[45]:


from math import sqrt


# In[46]:


lm = LinearRegression()
lm.fit(x_train,y_train)
y_pred_train = lm.predict(x_train)
q = r2_score(y_train,y_pred_train)

n = len(y_train)
k = x_train_pc.shape[1]

adj_r2 = 1-((1- r2_score(y_train,y_pred_train))*(n-1)/(n-k-1) )
rmse = sqrt(mean_squared_error(y_train,y_pred_train))

print('value of r2 is',{q})
print('value of adj_r2 is',{adj_r2})
print('value of rmse is',{rmse})


# In[47]:


lm = LinearRegression()
lm.fit(x_test,y_test)
y_pred_test = lm.predict(x_test)
q = r2_score(y_test,y_pred_test)

n = len(y_test)
k = x_test_pc.shape[1]

adj_r2 = 1-((1- r2_score(y_test,y_pred_test))*(n-1)/(n-k-1) )
rmse = sqrt(mean_squared_error(y_test,y_pred_test))

print('value of r2 is',{q})
print('value of adj_r2 is',{adj_r2})
print('value of rmse is',{rmse})


# In[48]:


import matplotlib.pyplot as plt

# Assuming you have the R2 and adjusted R2 values stored in variables
r2_score = 0.715287177788323
adjusted_r2_score = 0.6985393647170479
# Plotting the R2 and adjusted R2 scores
labels = ['R2', 'Adjusted R2']
scores = [r2_score, adjusted_r2_score]

plt.bar(labels, scores)
plt.ylim(0, 1)  # Adjust the y-axis limits according to your score range
plt.title('Accuracy Scores')
plt.ylabel('Score')
plt.show()


# In[49]:


plt.figure(figsize=(8,4))
plt.scatter(y_test,y_pred_test)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[56]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

knn = KNeighborsRegressor(n_neighbors=15)

knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
r2_knn = r2_score(y_test, y_pred_knn)
rmse = sqrt(mean_squared_error(y_test, y_pred_knn))

print('Value of r2 is', r2_knn)
print('Value of rmse is', rmse)


# In[58]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

dt = DecisionTreeRegressor(max_depth=5, random_state=44, min_samples_split=12)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
r2_dt = r2_score(y_test, y_pred_dt)
rmse = sqrt(mean_squared_error(y_test, y_pred_dt))

print('Value of r2 is', r2_dt)
print('Value of rmse is', rmse)


# In[63]:


from sklearn.svm import SVR

svm_1 = SVR(kernel ='poly') 
svm_1.fit(x_train, y_train)
y_pred_svm = svm_1.predict(x_test)
r2_svm_1 = r2_score(y_test, y_pred_svm)
rmse = sqrt(mean_squared_error(y_test, y_pred_svm))

print('Value of r2 is', r2_svm_1)
print('Value of rmse is', rmse)


# In[61]:


from sklearn.model_selection import GridSearchCV
param_grid={'gamma':[0.1,0.01,0.5,0.7,1], 'C':[0.1,1,10,20,15],
           'kernel':['rbf','sigmoid','linear']}
svr=SVR()
grid=GridSearchCV(svr,param_grid,cv=20)
grid.fit(x_train,y_train)
grid.best_params_


# In[64]:


from sklearn.svm import SVR

svm = SVR(kernel ='linear',C = 20, gamma = 0.1)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)
r2_svm = r2_score(y_test, y_pred_svm)
rmse = sqrt(mean_squared_error(y_test, y_pred_svm))

print('Value of r2 is', r2_svm)
print('Value of rmse is', rmse)


# In[54]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

rt = RandomForestRegressor(max_depth=5, random_state=44, min_samples_split=12, n_estimators=30)
rt.fit(x_train, y_train)
y_pred_rt = rt.predict(x_test)
r2_rt = r2_score(y_test, y_pred_rt)
rmse = sqrt(mean_squared_error(y_test, y_pred_rt))

print('Value of r2 is', r2_rt)
print('Value of rmse is', rmse)


# # Conclusion

# """K-Nearest Neighbors (KNN) and Linear Regression emerged as the top-performing models, 
# with KNN providing the highest accuracy. These models are recommended for pricing decisions.
# Further optimization and feature engineering may enhance predictive capabilities. 
# Stakeholders can confidently rely on these models to estimate house prices effectively."""

# """The linear regression model exhibits promising performance based on the provided accuracy metrics.
# With a training R-squared of 0.6906 and testing R-squared of 0.7153, it effectively explains a substantial portion of 
# the target variable's variance. Furthermore, the adjusted R-squared values of 0.6863 for training and 0.6985 for
# testing suggest that the model incorporates relevant predictors without undue complexity. Additionally, 
# the lower RMSE on the testing data (767,684.86) compared to training (1,006,984.24) indicates improved generalization 
# capabilities. While these results are positive, further domain-specific analysis and fine-tuning may be considered 
# to minimize prediction errors and optimize model performance."""

# # Generalization

# """The model generalizes well to unseen data, indicating that it is not overly tailored to the training dataset. 
# This means it can provide reliable price predictions for new or future properties."""

# #                                     """Thank You for coming"""

# In[ ]:




