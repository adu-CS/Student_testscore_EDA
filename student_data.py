#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




# In[16]:


data=pd.read_csv("C:\\Users\\adars\\Downloads\\student_extended_ml_dataset2.csv")


# In[5]:


data.head()


# In[6]:


data.tail()
#size:5000 values


# In[7]:


data.nunique()


# In[12]:


data.columns


# In[11]:


data.isnull().sum()


# In[15]:


description = data[['IQ', 'Physics_Marks']].describe()
print(description)


# In[19]:


data['Total_Marks']=data['Physics_Marks']+data['Chemistry_Marks']+data['Math_Marks']
#adding extra column-'Total_Marks'


# In[36]:


data.head()


# In[42]:


data['Has_Part_Time_Job'] = data['Has_Part_Time_Job'].map({True: 1, False: 0})


# In[46]:


data.head()
#CONVERTS BOOLEAN VARIABLE 'Has_Part_Time_Job' to BINARY VALUES


# In[39]:


#juz removing non-numeric values to make it compatible for Heatmap 
student = data.drop(['Name', 'Gender', 'Study_Hours_Group'], axis=1)


# In[40]:


sns.heatmap(student, annot=True, cmap='coolwarm')
plt.xlabel('Hours_Studied')
plt.ylabel('Total_Marks')
plt.title('mark_vs_hrs')

#display the heatmap
plt.show()


# In[41]:


sns.pairplot(student)


# In[32]:


sns.relplot(x='Hours_Studied', y='Total_Marks', data=student)
#We cannot find any plausible inferences from this,let's try other ways and variables to examine


# In[47]:


sns.relplot(x='Has_Part_Time_Job', y='Total_Marks', data=student)


# In[64]:


# Extract the independent variable (X) and dependent variable (y)
X = data[['IQ']]
y = data['Total_Marks']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Get the model parameters
slope = model.coef_[0]
intercept = model.intercept_

# Predict the Total_Marks values using the model
predicted_total_marks = model.predict(X)

# Create a scatter plot of IQ vs. Total_Marks
plt.scatter(X, y, label='Data Points')
plt.plot(X, predicted_total_marks, color='red', linewidth=3, label='Linear Regression Line')
plt.xlabel('IQ')
plt.ylabel('Total Marks')
plt.title('Linear Regression: IQ vs. Total Marks')
plt.legend()
plt.grid(True)

# Print the model parameters
print(f'Slope (Coefficient): {slope}')
print(f'Intercept: {intercept}')

# Show the plot
plt.show()


# In[69]:


X = data[['Physics_Marks']]
y = data['Math_Marks']


model = LinearRegression()


model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

predicted_math_marks = model.predict(X)

# Create a scatter plot of Physics_Marks vs. Math_Marks
plt.scatter(X, y, label='Data Points')
plt.plot(X, predicted_math_marks, color='red', linewidth=3, label='Linear Regression Line')
plt.xlabel('Physics Marks')
plt.ylabel('Math Marks')
plt.title('Linear Regression: Physics Marks vs. Math Marks')
plt.legend()
plt.grid(True)

print(f'Slope (Coefficient): {slope}')
print(f'Intercept: {intercept}')


plt.show()


# In[70]:


X = data[['Hours_Studied']]
y = data['Total_Marks']


model = LinearRegression()

model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_


predicted_math_marks = model.predict(X)

#SCatter plot Hours_Studied vs Tot_marks
plt.scatter(X, y, label='Data Points')
plt.plot(X, predicted_math_marks, color='red', linewidth=3, label='Linear Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Total Marks')
plt.title('Linear Regression: Hours studied vs. Total Marks')
plt.legend()
plt.grid(True)


print(f'Slope (Coefficient): {slope}')
print(f'Intercept: {intercept}')

# Show the plot
plt.show()


# In[71]:


import pandas as pd



# Calculate the median for 'Total_Marks' across the entire dataset
median_total_marks = data['Total_Marks'].median()

# Filter the data where 'Has_Part_Time_Job' is False
no_part_time_job_data = data[data['Has_Part_Time_Job'] == False]

# Calculate the mean of 'Total_Marks' for this filtered subset
mean_total_marks_no_part_time_job = no_part_time_job_data['Total_Marks'].mean()

# Check if 'Total_Marks' is above the calculated median for students without part-time jobs
above_median = mean_total_marks_no_part_time_job > median_total_marks

# Print the results
print(f'Median Total Marks: {median_total_marks}')
print(f'Mean Total Marks for No Part-Time Job: {mean_total_marks_no_part_time_job}')
print(f'Is Mean Total Marks for No Part-Time Job above Median: {above_median}')


# In[ ]:


#No satisfactory inferences from data.
#Tried analyzing various factors; univariate, multivariate analysis with help of Scatter plots, heatmaps.
#Linear Regression model was used to look for trend between few bivariate data.

