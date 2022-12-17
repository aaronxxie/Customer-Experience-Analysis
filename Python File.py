#!/usr/bin/env python
# coding: utf-8

# # Customer Experence Analysis based on Shipping Dataset
# ### Data Analytics Project (Python, Power BI)
# Aaron Xie
# ___
# ### The Problem
# Customer rating is a crucial metric to show how satisfied customers are with a business. Therefore, this project examinates the relationship between customer rating and the shipping features. What are the potential drawbacks of low customer ratings?
# 
# #### The Questions
# Shipping performance:
# * What are the total cost and shipping amount?
# * Which block/department has the highest shipping amount?
# * What is the most preferred ship mode?
# * Are the products gender neutral?
# * How much does the products usually cost?
# * How does the business offer discount based on cost?
# 
# Customer experience:
# * What are the average customer ratings?
# * Which customer feature is related to customer ratings?
# * Does the on time rate matter to customer ratings?
# * Can increasing discount offered raise customer ratings?
# * How many customers care calls does the business usually receive?
# * Does more care call mean bad experience(bad ratings)?
# * Does high prior purchases mean good experience(good ratings)?
# 
# 
# #### The Goal
# * Find the drawbacks that the business can improve to increase customer ratings.
# * Make strategies to overcome these drawbacks and raise the customer ratings by 20%

# ### Data Collection/Preparation
# This project uses a fictional dataset from Kaggle; check out the website to see its documentation.
# https://www.kaggle.com/datasets/prachi13/customer-analytics

# ### Data Processing/Cleaning

# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import scipy

plt.style.use("ggplot")
rcParams['figure.figsize'] = (12,  6)


# In[3]:


# import the dataset
shipping_df = pd.read_csv(r"C:\Users\zong0\OneDrive\Documents\Programming\My Project\Shipping EDA\e_commerce_shipping.csv")


# #### Dataset Basic Info

# In[4]:


shipping_df.head()


# In[5]:


shipping_df.shape


# In[6]:


shipping_df.describe()


# In[7]:


shipping_df.info()


# In[8]:


# check duplicate rows
shipping_df.duplicated().sum()


# Great! The dataset does not have missing values or duplicated rows

# In[9]:


# ID is just primary key, set it to index
shipping_df = shipping_df.set_index('ID')


# ### Data Analyses

# #### Understanding Variables

# ##### Categorical Variables

# Attention: the column 'Reached.on.Time_Y.N' is categorial variable, although it has int64 Dtype. So is the target column 'Customer_rating,' because it is ordinal

# In[10]:


# create a list of categorical variable
categorical_variables = ['Reached.on.Time_Y.N','Customer_rating']
for col in list(shipping_df.columns):
    if shipping_df[col].dtypes == object: 
        categorical_variables += [col] 
categorical_variables


# In[11]:


# Check the percentage of all categories for all categorical variables
for col in categorical_variables:
    print(shipping_df[col].value_counts(normalize = True))


# In[12]:


# Visualize the previous percentage (optional)

sns.set(font_scale=1.5)
fig, ax = plt.subplots(6, figsize=(30, 30))
for elem, i in zip(categorical_variables, range(6)):
        shipping_df[elem].value_counts().plot(kind='pie', ax=ax[i], autopct='%1.1f%%', title = elem, ylabel='')
fig.tight_layout()
plt.show()


# **The analysis on categorical variables indicates:**
# * The on time rate of the business is only 59.7%, which is relatively low; the business needs to improve shipping speed.
# * Warehouse Section F ships most of the products; the business needs to examinate whether they have enough labors in this section.
# * Most products are shipped by ship; this may be the reason for delay.
# * The products of the company is gender neutral. 
# * Customer Rating is nearly uniformly distributed, which is unusual. The business may have equal good and bad features. 

# Note: these speculations are based on this dataset; **if this dataset is just a sample dataset**, the project needs to add more tests to see whether this sample dataset can represent the population dataset.

# ##### Numeric Variables

# In[13]:


numeric_variables = []
for col in list(shipping_df.columns):
    if shipping_df[col].dtypes == np.int64: # note: don't just write 'int64'
        numeric_variables += [col] 
numeric_variables.remove('Reached.on.Time_Y.N')
numeric_variables.remove('Customer_rating')
numeric_variables


# In[14]:


# Describe the distributions
for elem in numeric_variables:
    print(shipping_df[elem].describe())


# In[15]:


# Check skewness and kurtosis
for elem in numeric_variables:
    print(f"{elem} Skewness: {shipping_df[elem].skew()}")
    print(f"{elem} Kurtosis: {shipping_df[elem].kurt()}")


# In[16]:


# Plot the histgram of numeric variables
sns.set(font_scale=3)
fig, ax = plt.subplots(5, figsize=(30, 100))
for elem, i in zip(numeric_variables, range(5)):
    shipping_df[elem].plot(kind='hist', ax=ax[i], title = elem)


# **The analysis on categorical variables indicates:**
# * The average customer care call is 4 times, which is frequent. This phenomenon indicates following possible problems: customer care provided has low quality and does not solve customers' problems; the business' system does not function well to fulfill all customers' needs; this is a business with high customization, yet the business has not built those customization into its website/app.
# * The prior_purchases is right skewed. It is more likely for those who have bought the products for 2 to 4 times to buy again. After that, they will cease to need these products. The products are not necessity; the business needs to diversify its products.
# * Mostly the company only has small discounts; it can consider increase the discount amount to attract more businesses.

# #### Understanding the Relationships between Variables

# In[74]:


# Skim the relationships with a scatterplot
sns.set(font_scale=1.5)
sns.pairplot(shipping_df);


# Some insights from scatterplots:
# * The products with high weights (>4000gms) only have lower discount offered
# * The products with medium weights (between 2000 and 4000gms) always cost higher (at least 180)
# * As for target 'Customer_rating,' it appears to have no relationship with other variables. Need further investigation.
# * The business offers more small discounts; they offers more discounts on products with medium costs.

# ##### Understanding relationship between 'Customer_rating' and all numeric variables

# Using **spearman correlation** because 'Customer_rating' is ordinal variable.

# In[18]:


# Examinate the spearman correlation and p-values between all variables and customer_rating
for col in numeric_variables:
        if col != 'Customer_rating':
            corr, p_value = scipy.stats.spearmanr(shipping_df[col], shipping_df['Customer_rating'])
            print(f'{col} correlation: {round(corr,3)}; p_value: {round(p_value,3)}')


# Given that none of these p-values is less than 0.05, the correlation values are not considered statistically significant; even if they are significant, the correlation values are too close to 0, which represents very weak correlation between the rank of two variables.

# Using **boxplots**

# In[19]:


sns.set(font_scale=3)
fig, ax = plt.subplots(len(numeric_variables), figsize=(30, 100))

for col, i in zip(numeric_variables, range(len(numeric_variables))):
        sns.boxplot(x="Customer_rating", y= col, ax=ax[i], data=shipping_df)
#plt.show()


# Unfortunately, the box plots within the same subplot appear to be very similar, which indicates weak association between these numeric variables and the target categorical variable Customer_rating

# ##### Understanding relationship between 'Customer_rating' and other categorical variables

# Use Contingency Table, Chi-Square Test, and Cramer's V
# * Reference: https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792

# In[57]:


def two_categorical(col1, col2):
    contingency_table = pd.crosstab(col1, col2)
    matrix = contingency_table.values
    chi2, pval= scipy.stats.chi2_contingency(matrix)[0:2]
    size = matrix.sum()
    phi2 = chi2/size
    row, col = matrix.shape
    phi2corr = max(0, phi2 - (col-1)*(row-1)/(size-1))
    rowcorr = row - (row - 1)**2/(size - 1)
    colcorr = col - (col - 1)**2/(size - 1)
    cramers_v = np.sqrt(phi2corr/min((colcorr - 1),(rowcorr - 1)))
    print(contingency_table)
    print(f'Chi-square: {round(chi2, 3)}, P-Value: {round(pval, 3)}, Cramer\'s V: {cramers_v}')


# In[61]:


for col in categorical_variables:
    if col != 'Customer_rating':
        print(two_categorical(shipping_df['Customer_rating'], shipping_df[col]))


# Huge p-value and 0 cramer's V entail that there is no association between 'Customer_rating' and other categorical variables. It is questionalbe that whether this fictional dataset was poorly constructed or just incomplete. There must be other reasons for the business to have a low rating. 

# Given that this dataset did not correspond to my expection, I choose to combine the analysis with my experience in the field to emphasize following variables in my report:
# * 'Reached.on.Time_Y.N' 
# * 'Customer_care_calls': having more calls means customer service cannot solve customers' problems efficiently
# * 'Prior_purchases': repeated purchases indicate that customer approves the products or services of the business
# * 'Discount_offered': A discount is always favorable

# ### Data Sharing

# Sharing data by using 2 Power BI Dashboards.

# #### Dashboard 1: Overall Shipping Performance

# ![Shipping%20Analysis%20New-1.jpg](attachment:Shipping%20Analysis%20New-1.jpg)

# Answering the questions:
# - What are the total cost and shipping amount? 
#     - $2.31M and 10,999
# - Which block/department has the highest shipping amount? 
#     - F
# - What is the most preferred ship mode? 
#     - Ship
# - Are the products gender neutral? 
#     - Yes
# - How much does the products usually cost? 
#     - Mostly they cost from around 130 to around 270.
# - How does the business offer discount based on cost? 
#     - The business offers more small discounts; they offers more discounts on products with medium costs (from around 130 to around 270).

# Note: by clicking the cube slicers to the left, we can answer these questions by each block/department.

# #### Dashboard 2: Customer Experience

# ![Shipping%20Analysis%20New-2.jpg](attachment:Shipping%20Analysis%20New-2.jpg)

# - What are the average customer ratings?
#     - 2.99
# - Which customer feature is related to customer ratings?
#     - According to earlier analysis, none is related. However, I include some features in the dashboard to further investigate.
# - Does the on time rate matter to customer ratings?
#     - By clicking "YES" on the upper left, we see the average rating is 3.01, which is 0.04 higher than the rating 2.97 when clicking "NO". It seems matter, but the 0 cramer's V value we got earlier indicates that this is by chance.
# - Can increasing discount offered raise customer ratings?
#     - No, according to the scatterplot on lower right corner, these two variables have no pattern.
# - How many customers care calls does the business usually receive?
#     - Usually 3 to 5.
# - Does more care call mean bad experience(bad ratings)?
#     - No. By clicking each column of the customer call frequency chart, we can see that although the 2 calls column has the highest rating 3.07, the ratings doesn't necessarily decrease as the calls increase. Also, they did not pass the correlation test.
# - Does high prior purchases mean good experience(good ratings)?
#     - No. Other than the 10 prior purchases column, which has a higher score 3.2. Others also show no pattern.

# ### Action Suggestions

# Based on the analysis, the dataset is very unreliable. To draw better conclusions, the business first needs to improve their data gathering systems. This is crucial. Otherwise the business needs to rely on third party data and just makes strategies based on the management team's experience and intuitions.

# In[ ]:




