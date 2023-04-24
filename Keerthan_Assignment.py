#!/usr/bin/env python
# coding: utf-8

# # Method-1 of Preprocessing

# In[208]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

preprocessed = pd.read_csv(r"C:\Users\pc\Desktop\Data Science A2\KEERTHAN\Stress-Predict-Dataset-main\Processed_data\Improved_All_Combined_hr_rsp_binary.csv")


# In[209]:


preprocessed.head()


# In[214]:


def Data_Reader(x):
    
    #DATA READING IN LOOP
    """ Reading data folder by folder in for loop by manual string intervention and if loop for x<10 because it shoudl
    generate values like 01,02,03... and after 9 it will generate 10,11,12..."""
    
    if x<10:
        ACC = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/ACC.csv')
        BVP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/BVP.csv')
        EDA = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/EDA.csv')
        HR = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/HR.csv')
        IBI = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/IBI.csv')
        tags = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/tags_S0'+str(x)+'.csv')
        TEMP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/TEMP.csv')
    else:
        ACC = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/ACC.csv')
        BVP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/BVP.csv')
        EDA = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/EDA.csv')
        HR = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/HR.csv')
        IBI = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/IBI.csv')
        tags = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/tags_S'+str(x)+'.csv')
        TEMP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/TEMP.csv')
    
    
    """We need to create a time attribute where each of the data is recorded, the column name of each dataframe attribute
    is the start time + 1 of the attribute"""
    ACC_time = int(float(ACC.columns[0])) - 1
    BVP_time = int(float(BVP.columns[0])) - 1
    EDA_time = int(float(EDA.columns[0])) - 1
    HR_time = int(float(HR.columns[0])) - 1
    IBI_time = int(float(IBI.columns[0])) - 1
    TEMP_time = int(float(TEMP.columns[0])) - 1
    
    """Dropping the first row value of all the dataframe attribute because the value is all same """
    
    ACC = ACC.drop(ACC.index[0:1]).reset_index(drop = True)
    BVP = BVP.drop(BVP.index[0:1]).reset_index(drop = True)
    EDA = EDA.drop(EDA.index[0:1]).reset_index(drop = True)
    HR = HR.drop(HR.index[0:1]).reset_index(drop = True)
    IBI = IBI.drop(IBI.index[0:1]).reset_index(drop = True)
    TEMP = TEMP.drop(TEMP.index[0:1]).reset_index(drop = True)
    
    """Creating new variable called loop which is generates the value if the lenght of the attribute"""
    
    ACC['loop'] = np.arange(ACC.shape[0])
    BVP['loop'] = np.arange(BVP.shape[0])
    EDA['loop'] = np.arange(EDA.shape[0])
    HR['loop'] = np.arange(HR.shape[0])
    IBI['loop'] = np.arange(IBI.shape[0])
    TEMP['loop'] = np.arange(TEMP.shape[0])
    
    """Fucntion to add the arange length attribute with the time for each row in seconds incremental"""
    def time(x,y):
        return x + y
    
    """applying the function"""
    ACC['Time(sec)'] = np.vectorize(time)(ACC['loop'], ACC_time)
    BVP['Time(sec)'] = np.vectorize(time)(BVP['loop'], BVP_time)
    EDA['Time(sec)'] = np.vectorize(time)(EDA['loop'], EDA_time)
    HR['Time(sec)'] = np.vectorize(time)(HR['loop'], HR_time)
    IBI['Time(sec)'] = np.vectorize(time)(IBI['loop'], IBI_time)
    TEMP['Time(sec)'] = np.vectorize(time)(TEMP['loop'], TEMP_time)
    
    """dropping loop attribute as it is not longer necessary"""
    ACC.drop('loop',axis = 1,inplace = True)
    BVP.drop('loop',axis = 1,inplace = True)
    EDA.drop('loop',axis = 1,inplace = True)
    HR.drop('loop',axis = 1,inplace = True)
    IBI.drop('loop',axis = 1,inplace = True)
    TEMP.drop('loop',axis = 1,inplace = True)
    
    #Renaming each dataframe 
    """Renaming the columns to our fitting so we can merge all the files under one name"""
    ACC.rename({ACC.columns[0]:'accelerometer_X',ACC.columns[1]:'accelerometer_Y',ACC.columns[2]:'accelerometer_Z'},axis = 1,inplace = True)
    BVP.rename({BVP.columns[0]:'BVP'},axis = 1,inplace = True)
    EDA.rename({EDA.columns[0]:'EDA'},axis = 1,inplace = True)
    HR.rename({HR.columns[0]:'heart_rate'},axis = 1,inplace=True)
    IBI.rename({IBI.columns[0]:'IBI_0',IBI.columns[1]:'IBI_1'},axis = 1,inplace = True)
    TEMP.rename({TEMP.columns[0]:'temp'},axis = 1,inplace = True)
    
    
    final_1 = ACC.merge(BVP, on = 'Time(sec)',how = 'outer').merge(EDA, on = 'Time(sec)',how = 'outer').merge(HR, on = 'Time(sec)',how = 'outer').merge(IBI, on = 'Time(sec)',how = 'outer').merge(TEMP, on = 'Time(sec)',how = 'outer')
    
    """Finally merging with preprocessed data on time(sec) to get the label attribute"""
    final_1 = final_1.merge(preprocessed[['Label','Time(sec)']],on = 'Time(sec)',how = 'inner')
    
    final_1.fillna(method='ffill', inplace=True)
    final_1.fillna(method='bfill', inplace=True)

"""There are 35 subjects data which means i have to loop through 35 folders to read the data and concat each of it after processing"""
"""Left Subject 01 for testing"""
final_df1 = pd.DataFrame()
for i in range(2,36):
    Data_Reader(i)
    final_df1 = pd.concat([final_df1,final_1],ignore_index = True)
    if i<10:
        print('Folder S0'+str(i),' is extracted/Preprocessed and merged to single csv')
    else:
        print('Folder S'+str(i),' is extracted/Preprocessed and merged to single csv')


# In[217]:


final_df1.shape


# In[218]:


final_df1.head()


# In[219]:


final_df1.to_csv('Merged.csv')


# # Basic EDA

# In[220]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# In[221]:


final_df1 = pd.read_csv('Merged.csv')


# In[222]:


"""Checking null value of each individual columns"""
final_df1.isnull().sum()


# In[223]:


"""Checking Duplicated value of entire dataframe"""
final_df1.duplicated().sum()


# In[224]:


"""Checking value count of 0's and 1's """

"""From this description we get to know that the data is imbalanced with more number of 0's and less number of 1's
But this can be solved by downsampling and upsampling"""
final_df1['Label'].value_counts()


# In[225]:


sns.countplot(final_df1['Label'])


# In[226]:


#Dropping columns which are not required for dependent variable to classify
final_df1.drop({"Unnamed: 0"},axis = 1,inplace = True)


# In[228]:


import datetime

final_df1['datetime'] = pd.to_datetime(final_df1['Time(sec)'], unit='s')


# In[229]:


final_df1.head()


# In[241]:


fig, ax = plt.subplots()

# Plot the data as a time series
ax.plot(final_df1.datetime, final_df1['BVP'])

# Set the axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot of BVP')

# Display the plot
plt.show()


# In[232]:


fig, ax = plt.subplots()

# Plot the data as a time series
ax.plot(final_df1.datetime, final_df1['accelerometer_X'])
ax.plot(final_df1.datetime, final_df1['accelerometer_Y'])
ax.plot(final_df1.datetime, final_df1['accelerometer_Z'])

# Set the axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot of Accelerometer')

# Display the plot
plt.show()


# In[240]:


fig, ax = plt.subplots()

# Plot the data as a time series
ax.plot(final_df1.datetime, final_df1['EDA'])

# Set the axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot of EDA')

# Display the plot
plt.show()


# In[238]:


fig, ax = plt.subplots()

# Plot the data as a time series
ax.plot(final_df1.datetime, final_df1['heart_rate'])

# Set the axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot of Heart rate')

# Display the plot
plt.show()


# In[239]:


fig, ax = plt.subplots()

# Plot the data as a time series
ax.plot(final_df1.datetime, final_df1['temp'])

# Set the axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot of temperature')

# Display the plot
plt.show()


# In[242]:


fig, ax = plt.subplots()

# Plot the data as a time series
ax.plot(final_df1.datetime, final_df1['IBI_0'])
ax.plot(final_df1.datetime, final_df1['IBI_1'])

# Set the axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot of Accelerometer')

# Display the plot
plt.show()


# # Downsampling

# In[250]:


"""Separate Zeros and Ones then downsampling the 0's and concat them together to balance the data"""
zero_final_df1 = final_df1[final_df1['Label'] == 0]
one_final_df1 = final_df1[final_df1['Label'] == 1]

"""Randomly selecting 37000 0's data"""
zero_final_df1 = zero_final_df1.sample(108868)


# In[251]:


zero_final_df1.shape,one_final_df1.shape


# In[252]:


"""Concat zeros and ones after downsampling"""
down_final_df1 = pd.concat([zero_final_df1,one_final_df1],ignore_index = True)


# In[253]:


down_final_df1.shape


# In[254]:


down_final_df1.head()


# In[255]:


down_final_df1['Label'].value_counts()


# In[256]:


sns.countplot(down_final_df1['Label'])


# In[257]:


sns.pairplot(down_final_df1)
plt.show()


# # Skewed Data

# In[258]:


"""Histogram plot helps us figure out which data attribute is skewed left or right.
Here for ex EDA is right skewed data and temp is left skewed data"""
down_final_df1.hist(figsize = (20,10))
plt.show()


# # Outlier

# In[259]:


"""Box plot helps us find the outlier which will be handled while modeling by z-score"""
plt.figure(figsize=(10,10))
sns.boxplot(data=down_final_df1)
plt.xticks(rotation=90)
plt.show()


# # Identifying highly posively correlated and highly negatively correlated attributes

# In[260]:


down_final_df1.corr()


# In[261]:


"""Heat map is a visual representation of correlation table"""
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(down_final_df1.corr(), linewidth=0.8,annot = True)
plt.show()


# # Upsamling

# In[265]:


"""Separate Zeros and Ones then upsampling the 1's and concat them together to balance the data"""
z_final_df1 = final_df1[final_df1['Label'] == 0]
o_final_df1 = final_df1[final_df1['Label'] == 1]


# In[266]:


"""Split data for fitting into smote"""
X = final_df1.drop({'Label','datetime'},axis = 1) #"""select all rows and first 8 columns which are the attributes"""
Y = final_df1['Label']   #"""select all rows and the 8th column which is the classification "Yes", "No" for diabeties"""
test_size = 0.30 #"""taking 70:30 training and test set"""
seed = 7  #"""Random numbmer seeding for reapeatability of the code"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[267]:


print("Before Upsampling 1's:- ",o_final_df1.shape[0])

"""Synthetic Minority Over Sampling Technique AKA SMOTE"""
sm = SMOTE()   
dependent, label = sm.fit_resample(X_train, y_train.ravel())
print("After UpSampling 1's:- {}".format(sum(label==1)))


# In[268]:


up_final_df1 = dependent.copy()
up_final_df1['Label'] = label


# In[269]:


up_final_df1.head()


# In[270]:


up_final_df1['Label'].value_counts()


# In[271]:


sns.countplot(up_final_df1['Label'])


# # skewed data

# In[272]:


"""Histogram plot helps us figure out which data attribute is skewed left or right.
Here for ex EDA is right skewed data and temp is left skewed data"""
up_final_df1.hist(figsize = (20,10))
plt.show()


# # Outlier

# In[273]:


"""Box plot helps us find the outlier which will be handled while modeling by z-score"""
plt.figure(figsize=(10,10))
sns.boxplot(data=up_final_df1)
plt.xticks(rotation=90)
plt.show()


# # Identifying highly posively correlated and highly negatively correlated attributes

# In[274]:


up_final_df1.corr()


# In[275]:


"""Heat map is a visual representation of correlation table"""
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(up_final_df1.corr(), linewidth=0.8,annot = True)
plt.show()


# # Splitting the Data into Training and testing set

# In[276]:


from sklearn.model_selection import train_test_split


# In[283]:


X = down_final_df1.drop({'Label','datetime','Time(sec)'},axis = 1)
y = down_final_df1['Label']


# In[284]:


#Splitting training and testing into 805 and 20% ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[285]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[286]:


data = [X_train.shape[0],y_train.shape[0],X_test.shape[0],y_test.shape[0]]
labels = ['X Training', 'Y Training', 'X testing', 'Y Testing']

# Set up the plot
fig, ax = plt.subplots()

# Plot the bars
ax.bar(labels, data)

# Add labels and adjust the layout
ax.set_xlabel('Split Ratio')
ax.set_ylabel('Value')
plt.tight_layout()

# Show the plot
plt.show()


# # Model Building

# # LR (Logistic Regression)

# In[287]:


from sklearn.linear_model import LogisticRegression


# In[288]:


LR = LogisticRegression(random_state=42).fit(X_train, y_train)
LR_predict = LR.predict(X_test)


# In[289]:


from sklearn.metrics import classification_report


# In[290]:


print(classification_report(y_test,LR_predict))


# # Random Forest (RF)

# In[291]:


from sklearn.ensemble import RandomForestClassifier


# In[292]:


RF = RandomForestClassifier().fit(X_train, y_train)
RF_predict = RF.predict(X_test)


# In[293]:


print(classification_report(y_test,RF_predict))


# # Naive Bayes (NB)

# In[294]:


from sklearn.naive_bayes import GaussianNB


# In[295]:


GNB = GaussianNB().fit(X_train, y_train)
GNB_predict = GNB.predict(X_test)


# In[296]:


print(classification_report(y_test,GNB_predict))


# # ADABOOST Classifier

# In[297]:


from sklearn.ensemble import AdaBoostClassifier


# In[299]:


ADA = AdaBoostClassifier().fit(X_train, y_train)
ADA_predict = ADA.predict(X_test)


# In[300]:


print(classification_report(y_test,ADA_predict))


# # GradientBoosting Classifier

# In[301]:


from sklearn.ensemble import GradientBoostingClassifier


# In[302]:


GBC = GradientBoostingClassifier().fit(X_train, y_train)
GBC_predict = GBC.predict(X_test)


# In[303]:


print(classification_report(y_test,GBC_predict))


# # Model Comparision

# In[306]:


from sklearn.metrics import accuracy_score

# Create a dictionary to store the models
models = {'RANDOM FOREST': RandomForestClassifier(),
          'LOGISTIC REGRESSION': LogisticRegression(random_state=42),
         'ADA BOOST':AdaBoostClassifier(),
         "GRADIENT BOOSTING":GradientBoostingClassifier(),
         "NAIVE BAYES:":GaussianNB()}

# Train and evaluate each model
accuracies = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f'{name}: {accuracy}')

# Plot the accuracy scores
plt.bar(np.arange(len(models)), accuracies)
plt.xticks(np.arange(len(models)), models.keys())
plt.title('Accuracy Scores of Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()


# # Predicting for Unseen Data which is Subject 1 S01.

# In[ ]:


#I have merged subject 2 - subject 35 which is s02-s35 and left s01 so i can train the models and predict on this subject as test


# In[323]:


x = 1

if x<10:
    ACC = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/ACC.csv')
    BVP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/BVP.csv')
    EDA = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/EDA.csv')
    HR = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/HR.csv')
    IBI = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/IBI.csv')
    tags = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/tags_S0'+str(x)+'.csv')
    TEMP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S0'+str(x)+'/TEMP.csv')
else:
    ACC = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/ACC.csv')
    BVP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/BVP.csv')
    EDA = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/EDA.csv')
    HR = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/HR.csv')
    IBI = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/IBI.csv')
    tags = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/tags_S'+str(x)+'.csv')
    TEMP = pd.read_csv('C:/Users/pc/Desktop/Data Science A2/KEERTHAN/Stress-Predict-Dataset-main/Raw_data/S'+str(x)+'/TEMP.csv')
    
    
"""We need to create a time attribute where each of the data is recorded, the column name of each dataframe attribute
is the start time + 1 of the attribute"""
ACC_time = int(float(ACC.columns[0])) - 1
BVP_time = int(float(BVP.columns[0])) - 1
EDA_time = int(float(EDA.columns[0])) - 1
HR_time = int(float(HR.columns[0])) - 1
IBI_time = int(float(IBI.columns[0])) - 1
TEMP_time = int(float(TEMP.columns[0])) - 1

"""Dropping the first row value of all the dataframe attribute because the value is all same """

ACC = ACC.drop(ACC.index[0:1]).reset_index(drop = True)
BVP = BVP.drop(BVP.index[0:1]).reset_index(drop = True)
EDA = EDA.drop(EDA.index[0:1]).reset_index(drop = True)
HR = HR.drop(HR.index[0:1]).reset_index(drop = True)
IBI = IBI.drop(IBI.index[0:1]).reset_index(drop = True)
TEMP = TEMP.drop(TEMP.index[0:1]).reset_index(drop = True)

"""Creating new variable called loop which is generates the value if the lenght of the attribute"""

ACC['loop'] = np.arange(ACC.shape[0])
BVP['loop'] = np.arange(BVP.shape[0])
EDA['loop'] = np.arange(EDA.shape[0])
HR['loop'] = np.arange(HR.shape[0])
IBI['loop'] = np.arange(IBI.shape[0])
TEMP['loop'] = np.arange(TEMP.shape[0])

"""Fucntion to add the arange length attribute with the time for each row in seconds incremental"""
def time(x,y):
    return x + y

"""applying the function"""
ACC['Time(sec)'] = np.vectorize(time)(ACC['loop'], ACC_time)
BVP['Time(sec)'] = np.vectorize(time)(BVP['loop'], BVP_time)
EDA['Time(sec)'] = np.vectorize(time)(EDA['loop'], EDA_time)
HR['Time(sec)'] = np.vectorize(time)(HR['loop'], HR_time)
IBI['Time(sec)'] = np.vectorize(time)(IBI['loop'], IBI_time)
TEMP['Time(sec)'] = np.vectorize(time)(TEMP['loop'], TEMP_time)

"""dropping loop attribute as it is not longer necessary"""
ACC.drop('loop',axis = 1,inplace = True)
BVP.drop('loop',axis = 1,inplace = True)
EDA.drop('loop',axis = 1,inplace = True)
HR.drop('loop',axis = 1,inplace = True)
IBI.drop('loop',axis = 1,inplace = True)
TEMP.drop('loop',axis = 1,inplace = True)

#Renaming each dataframe 
"""Renaming the columns to our fitting so we can merge all the files under one name"""
ACC.rename({ACC.columns[0]:'accelerometer_X',ACC.columns[1]:'accelerometer_Y',ACC.columns[2]:'accelerometer_Z'},axis = 1,inplace = True)
BVP.rename({BVP.columns[0]:'BVP'},axis = 1,inplace = True)
EDA.rename({EDA.columns[0]:'EDA'},axis = 1,inplace = True)
HR.rename({HR.columns[0]:'heart_rate'},axis = 1,inplace=True)
IBI.rename({IBI.columns[0]:'IBI_0',IBI.columns[1]:'IBI_1'},axis = 1,inplace = True)
TEMP.rename({TEMP.columns[0]:'temp'},axis = 1,inplace = True)


subject01 = ACC.merge(BVP, on = 'Time(sec)',how = 'outer').merge(EDA, on = 'Time(sec)',how = 'outer').merge(HR, on = 'Time(sec)',how = 'outer').merge(IBI, on = 'Time(sec)',how = 'outer').merge(TEMP, on = 'Time(sec)',how = 'outer')

subject01.fillna(method='ffill', inplace=True)
subject01.fillna(method='bfill', inplace=True)


# In[324]:


subject01


# In[325]:


subject01_predicted = RF.predict(subject01.drop('Time(sec)',axis = 1))


# In[327]:


subject01['Predicted'] = subject01_predicted


# In[330]:


subject01.head()


# In[329]:


subject01.to_csv('Test_predicted.csv')


# In[ ]:




