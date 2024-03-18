#!/usr/bin/env python
# coding: utf-8

# In[1]:


##!pip install districtdatalabs yellowbrick


# In[2]:


conda install -c districtdatalabs yellowbrick


# In[3]:


#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)


# In[4]:


#load dataset
data =pd.read_csv("marketing_campaign - marketing_campaign.csv")
data.head().T


# In[5]:


#assesing data
data.info()


# #### observations
# 
# * There are missing values in income
# * Dt_Customer that indicates the date a customer joined the database is not parsed as DateTime

# In[6]:


#drop missing values
data.dropna(inplace=True)


# In[7]:


# check for duplicates
print(data.duplicated().sum())


# In[8]:


data.columns


# In[9]:


#changing the date dtype to datetime object
data["Dt_Customer"]=pd.to_datetime(data["Dt_Customer"],format='mixed')


# In[10]:


type(data["Dt_Customer"][0])


# ### feature engineering
# 
# - Creating new features; "Customer_For","Age" and " is Parent" and "total spent"
# - modifying existing features; "education",
# - renmaing some features for more clarity
# - dropped off more redundant features

# In[11]:


#Created a feature "Customer_For"
#of the number of days the customers started to shop in the store relative to the last recorded date
days = []
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)

d1 = max(dates)# taking it the newest customer
for x in dates:
    delta = d1 - x
    days.append(delta)

data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"])


# In[12]:


data.columns


# In[13]:


#explore the material status and the education level features
data['Marital_Status'].value_counts()


# In[14]:


data['Education'].value_counts()


# In[15]:


data.columns


# In[16]:


#Feature Engineering
#Age of customer today
data["Age"]= 2021 - data['Year_Birth'] 

#Total spendings on various items
data["Spent_Amt"] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
    

#Deriving living situation by marital status"Alone"
data["Living_With"] = data['Marital_Status'].replace({"Married":"Partner","Together":"Partner","Single":"Alone","Divorced":"Alone","Widow":"Alone","Alone":"Alone","Absurd":"Alone","YOLO":"0"})
   
#Feature indicating total children living in the household
data["Children"] = data["Kidhome"] + data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = pd.to_numeric(data["Living_With"].replace({"Partner":2,"Alone":1})) + data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children > 0,1,0)

#Segmenting education levels in three groups
data['Education'] = data['Education'].replace({"Graduation":"Graduate","PhD":"PhD","Master":"PostGraduate","2n Cycle":"PostGraduate","Basic":"Undergraduate"})

#For clarity
data=data.rename(columns={'MntWines':"Wines", 'MntFruits':'Fruits',
       'MntMeatProducts':'MeatProducts','MntFishProducts':"FishProducts",'MntSweetProducts':"SweetProducts",
       'MntGoldProds':'GoldProds'})

#Dropping some of the redundant features
to_drop = ['Year_Birth','Marital_Status',"Kidhome","Teenhome",'Z_CostContact', 'Z_Revenue','Dt_Customer','ID']
data = data.drop(to_drop,axis = 1)


# In[17]:


data


# ### EXPLORATORY DATA ANALYIS
# - multivariate exploration to explore features relationships
# 

# In[18]:


#check descriptive statisitc
data.describe().T


# In[19]:


data.describe(include=(bool,object))


# In[20]:


print(data.isnull().sum())


# In[21]:


### visualizing the missing values
sns.heatmap(data.isnull(),annot=True,fmt="g")


# In[22]:


data1 = data.copy(deep=True)


# In[23]:


data1.columns


# In[24]:


# UNIVARIATE ANALYSIS FOR CUSTOMERS EDUCATIONAL QUALIFICATION 

ax = sns.countplot(x='Education', data=data1, order=data1['Education'].value_counts().index, color='skyblue')
ax.bar_label(ax.containers[0])
plt.title("Count of Customer Educational Qualification")
plt.show()


# In[25]:


# UNIVARIATE ANALYSIS FOR CUSTOMERS LIVING WITH SOMEONE

ax = sns.countplot(x='Living_With', data=data1, order=data1['Living_With'].value_counts().index, color='skyblue')
ax.bar_label(ax.containers[0])
plt.title("Count of Customer living with someone")
plt.show()


# In[26]:


# UNIVARIATE ANALYSIS FOR CUSTOMERS WHO ARE PARENTS

ax = sns.countplot(x='Is_Parent', data=data1, order=data1["Is_Parent"].value_counts().index, color='skyblue')
ax.bar_label(ax.containers[0])
plt.title("Count of Customer\s who are Parents")
plt.show()


# In[27]:


# UNIVARIATE ANALYSIS FOR CUSTOMERS FAMILY SIZE

ax = sns.countplot(x='Family_Size', data=data1, order=data1["Family_Size"].value_counts().index, color='skyblue')
ax.bar_label(ax.containers[0])
plt.title("Distribution of Customers Family size")
plt.show()


# In[28]:


# UNIVARIATE ANALYSIS FOR DISTRIBUTION OF DAYS SINCE THE CUSTOMERS LAST PURCHASE
plt.figure(figsize =(10,8))
sns.histplot(x='Recency', data=data1, color='skyblue')
plt.title("Distribution of Days Since Customer\'s Last Purchase")
plt.ylabel("Frequency")
plt.show()


# In[29]:


# UNIVARIATE ANALYSIS FOR CUSTOMER Recency
sns.boxplot(x="Recency",data=data1)
plt.title("Box plot visualization for customer\s Recency")
plt.show()


# In[30]:


# UNIVARIATE ANALYSIS FOR CUSTOMER INCOME 


plt.figure(figsize =(10,8))
sns.histplot(x='Income', data=data1,kde=True, color='skyblue')
plt.title("Customer income")
plt.ylabel("Customer income")
plt.show()


# In[31]:


# UNIVARIATE DISTRIBUTION OF CUSTOMER INCOME 
sns.boxplot(x="Income",data=data1)
plt.title("Box plot visualization for customer\s income")
plt.show()


# In[32]:


max_age=max(data1["Income"])
max_age


# In[33]:


min_age=min(data1["Income"])
min_age


# In[34]:


#UNIVARIATE ANALYSIS FOR CUSTOMER INCOME GROUP

def customer_income(income):
    if income <=223375:
        return "Low income <=223375"
    elif 223375 < income <= 445021.7:
        return "Middle income <=445021.7"
    else:
        return "High income <=666668.0"

data1["Customer_income_group"]=data1["Income"].apply(customer_income)

plt.figure(figsize=(10,8))

ax=sns.countplot(x="Customer_income_group",data=data1,order=data1["Customer_income_group"].value_counts().index)

ax.bar_label(ax.containers[0])

plt.title("Visualisation of Customer Income Group")

plt.show()


# In[35]:


# UNIVARIATE ANALYSIS FOR DISTRIBUTION OF DAYS SINCE THE CUSTOMERS LAST PURCHASE
plt.figure(figsize =(10,8))
sns.histplot(x='Spent_Amt', data=data1,kde=True, color='skyblue')
plt.title("Distribution of Toatal Amount Spent by the Customer")
plt.show()


# In[36]:


# UNIVARIATE DISTRIBUTION OF CUSTOMER INCOME 
sns.boxplot(x="Spent_Amt",data=data1)
plt.title("Box plot visualization for the Amount Spent By the Customer")
plt.show()


# In[ ]:





# In[37]:


max_age=max(data1["Age"])
max_age


# In[38]:


min_age=min(data1["Age"])
min_age


# In[39]:


def customer_agegroup(Age):
    if Age <= 45.6:
        return "Adult <=45.6"
    elif Age <= 67.2:
        return "Old Adult <=67.2"
    elif Age <= 88.8:
        return "Seniors <= 88.8"
    elif Age <= 110.4:
        return "Centenarians <=110.4"
    else:
        return "Supercentenerians <=132"

data1["Customer_agegroup"]=data1["Age"].apply(customer_agegroup)

plt.figure(figsize=(10,8))

ax=sns.countplot(x="Customer_agegroup",data=data1,order=data1["Customer_agegroup"].value_counts().index)

ax.bar_label(ax.containers[0])

plt.title("Visualisation of Customer Age Group")

plt.show()


# In[40]:


data1.columns


# In[41]:


#UNIVARIATE ANALYSIS FOR NUMBER OF PURCHASE MADE WITH A DISCOUNT
plt.figure(figsize=(10,8))
ax=sns.countplot(x='NumDealsPurchases',data=data1,order=data1['NumDealsPurchases'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Number of purchase made with a Discount")
plt.show()


# In[42]:


#UNIVARIATE ANALYSIS FOR NUMBER OF PURCHASE MADE THROUGH THE WEB
plt.figure(figsize=(10,8))
ax=sns.countplot(x='NumWebPurchases',data=data1,order=data1['NumWebPurchases'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Number of purchase made through the web")
plt.show()


# In[43]:


#UNIVARIATE ANALYSIS FOR NUMBER OF PURCHASE MADE USING A CATALOG
plt.figure(figsize=(10,8))
ax=sns.countplot(x='NumCatalogPurchases',data=data1,order=data1['NumCatalogPurchases'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Number of purchase made using the Catalog")
plt.show()


# In[44]:


#UNIVARIATE ANALYSIS FOR NUMBER OF PURCHASE MADE USING THE STORE
plt.figure(figsize=(10,8))
ax=sns.countplot(x='NumStorePurchases',data=data1,order=data1['NumStorePurchases'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Number of purchase made using the Store")
plt.show()


# In[45]:


#UNIVARIATE ANALYSIS FOR NUMBER OF WEBSITE VISIT PER MONTH
plt.figure(figsize=(10,8))
ax=sns.countplot(x='NumWebVisitsMonth',data=data1,order=data1['NumWebVisitsMonth'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Number of website   visit per month")
plt.show()


# In[46]:


data1.columns


# In[47]:


#UNIVARIATE ANALYSIS FOR CUSTOMERS WHO ACCEPTED THE OFFER IN THE 1ST CAMPAIGN
plt.figure(figsize=(10,8))
ax=sns.countplot(x='AcceptedCmp1',data=data1,order=data1['AcceptedCmp1'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Customer who accepted the offer in the 1st Campaign")

plt.show()


# In[48]:


#UNIVARIATE ANALYSIS FOR CUSTOMERS WHO ACCEPTED THE OFFER IN THE 2ND CAMPAIGN
plt.figure(figsize=(10,8))
ax=sns.countplot(x='AcceptedCmp2',data=data1,order=data1['AcceptedCmp2'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Customer who accepted the offer in the 2nd Campaign")

plt.show()


# In[49]:


#UNIVARIATE ANALYSIS FOR CUSTOMERS WHO ACCEPTED THE OFFER IN THE 3RD CAMPAIGN
plt.figure(figsize=(10,8))
ax=sns.countplot(x='AcceptedCmp3',data=data1,order=data1['AcceptedCmp3'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Customer who accepted the offer in the 3rd Campaign")

plt.show()


# In[50]:


#UNIVARIATE ANALYSIS FOR CUSTOMERS WHO ACCEPTED THE OFFER IN THE 4Th CAMPAIGN
plt.figure(figsize=(10,8))
ax=sns.countplot(x='AcceptedCmp4',data=data1,order=data1['AcceptedCmp4'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Customer who accepted the offer in the 4th Campaign")

plt.show()


# In[51]:


#UNIVARIATE ANALYSIS FOR CUSTOMERS WHO ACCEPTED THE OFFER IN THE 5TH CAMPAIGN
plt.figure(figsize=(10,8))
ax=sns.countplot(x='AcceptedCmp5',data=data1,order=data1['AcceptedCmp5'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Customer who accepted the offer in the 5th Campaign")

plt.show()


# In[52]:


# UNIVARIATE ANALYSIS FOR WINE PRODUCT PURCHASED
plt.figure(figsize=(10, 6))
sns.histplot(x='Wines', data=data, color='salmon',bins=20,kde=True)
plt.title('Proportion of Customers Buying Meat Products')
plt.xlabel('Wine Products Purchased')
plt.ylabel('Percentage')
plt.show()


# In[53]:


# UNIVARIATE ANALYSIS FOR FRUIT PRODUCT PURCHASED
plt.figure(figsize=(10, 6))
sns.histplot(x='Fruits', data=data, color='salmon',bins=20,kde=True)
plt.title('Proportion of Customers Buying Fruit Products')
plt.xlabel('Fruit Products Purchased')
plt.ylabel('Percentage')
plt.show()


# In[54]:


# UNIVARIATE ANALYSIS FOR MEAT PRODUCT PURCHASED
plt.figure(figsize=(10, 6))
sns.histplot(x='MeatProducts', data=data, color='salmon',bins=20,kde=True)
plt.title('Proportion of Customers Buying Meat Products')
plt.xlabel('Meat Products Purchased')
plt.ylabel('Percentage')
plt.show()


# In[55]:


# UNIVARIATE ANALYSIS FOR FISH PRODUCT PURCHASED
plt.figure(figsize=(10, 6))
sns.histplot(x='FishProducts', data=data, color='salmon',bins=20,kde=True)
plt.title('Proportion of Customers Buying Fish Products')
plt.xlabel('Fish Products Purchased')
plt.ylabel('Percentage')
plt.show()


# In[56]:


# UNIVARIATE ANALYSIS FOR SWEET PRODUCT PURCHASED
plt.figure(figsize=(10, 6))
sns.histplot(x='SweetProducts', data=data, color='salmon',bins=20,kde=True)
plt.title('Proportion of Customers Buying Sweet Products')
plt.xlabel('Sweet Products Purchased')
plt.ylabel('Percentage')
plt.show()


# In[57]:


# UNIVARIATE ANALYSIS FOR GOLD PRODUCT PURCHASED
plt.figure(figsize=(10, 6))
sns.histplot(x='GoldProds', data=data, color='salmon',bins=20,kde=True)
plt.title('Proportion of Customers Buying Gold Products')
plt.xlabel('Gold Products Purchased')
plt.ylabel('Percentage')
plt.show()


# In[58]:


# UNIVARIATE ANALYSIS FOR
plt.figure(figsize=(10, 6))
sns.histplot(x='Customer_For', data=data, color='salmon',bins=20,kde=True)
plt.title('Distribution of Customer Tenure')
plt.xlabel('Number of days')
plt.ylabel('Frequency')
plt.show()


# In[59]:


data1.columns


# In[60]:


# BIVARIATE ANALYSIS BETWEEN AGE GROUP AND AMOUNT SPENT
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
Customer_agegroup_SptAmt = data1.groupby('Customer_agegroup')['Spent_Amt'].sum().reset_index()

sns.barplot(x='Customer_agegroup', data=data1, y='Spent_Amt', ax=axs[0, 0])
axs[0, 0].set_title("Total Amount spent by Customer Age Group")
plt.xticks(rotation=15)

Customer_agegroup_SptAmt = data1.groupby('Customer_agegroup')['Income'].sum().reset_index()

sns.barplot(x='Customer_agegroup', data=data1, y='Income', ax=axs[0, 1])
axs[0, 1].set_title("Customer Income by Age Group")
plt.xticks(rotation=15)



SptAmt_fam_size = data1.groupby('Spent_Amt')['Family_Size'].sum().reset_index()

sns.scatterplot(x='Spent_Amt', data=data1, y='Family_Size', ax=axs[0, 2])
axs[0, 1].set_title("Amount Spent per family size")
plt.xticks(rotation=15)


SptAmt_fam_size = data1.groupby('Spent_Amt')['Customer_For'].sum().reset_index()

sns.scatterplot(x='Spent_Amt', data=data1, y='Customer_For', ax=axs[1,0 ])
axs[0, 1].set_title("Total Amount Spent per Customer")
plt.xticks(rotation=15)




SptAmt_fam_size = data1.groupby('Spent_Amt')['Income'].sum().reset_index()

sns.scatterplot(x='Spent_Amt', data=data1, y='Income', ax=axs[1,1])
axs[0, 1].set_title("Customer Income")
plt.xticks(rotation=15)


SptAmt_fam_size = data1.groupby('Spent_Amt')['Is_Parent'].sum().reset_index()

sns.scatterplot(x='Spent_Amt', data=data1, y='Is_Parent', ax=axs[1,2 ])
axs[0, 1].set_title("Amount Spent per family size")
plt.xticks(rotation=15)

















plt.show()


# In[61]:


# BIVARIATE ANALYSIS FOR AGE AND INCOME
plt.figure(figsize=(10,8))
sns.scatterplot(x='Age',data=data1,y='Income')
plt.show()


# In[62]:


# BIVARIATE ANALYSIS FOR FAMILY SIZE AND AMOUNT SPENT
plt.figure(figsize=(10,8))
sns.scatterplot(x='Family_Size',data=data1,y="Spent_Amt")
plt.show()


# ### Observation
# Upon conducting this analysis, a discernible trend emerges indicating a negative correlation between family size and the total amount spent. Notably, as the family size expands, there is a corresponding decrease in the overall amount spent. This observation sheds light on the potential influence of family size on consumer spending behavior, suggesting that larger families may exhibit more conservative spending habits. This insight can be valuable for marketing and promotional strategies, as it highlights the need for tailored approaches based on household characteristics.

# In[63]:


# BIVARIATE ANALYSIS FOR  EDUCATION AND WINE
plt.figure(figsize=(10,8))
sns.barplot(x='Education',data=data1,y='Wines')
plt.show()


# In[64]:


# BIVARIATE ANALYSIS FOR  NUMBER OF DEALS PURCHASED AND AMOUNTS SPENT
plt.figure(figsize=(10,8))
sns.scatterplot(x='NumDealsPurchases',data=data1,y='Spent_Amt')
plt.show()


# ## Observation
# Upon careful examination of the data, it becomes apparent that there exists a negative correlation between the amount spent and the number of deals purchased with a discount. Specifically, as the count of discounted deals increases, there is a corresponding decrease in the total amount spent. This negative relationship suggests that customers may be more inclined to limit their spending when taking advantage of discounted deals, impacting the overall revenue generated from such promotions. Understanding this correlation is essential for optimizing discount strategies and their impact on customer spending behavior.

# In[ ]:





# In[ ]:





# In[65]:


#multivariate Analysis- ANALYSING CUSTOMER AGE GROUP,EDUCATION AND AMOUNT SPENDT

plt.figure(figsize=(10,8))
sns.barplot(x='Customer_agegroup',data=data1,y='Spent_Amt',hue='Education')
plt.show()



# 
# ## Observation
# 
# ### Spending Patterns Across Age and Education Groups:
# Super centenarians (those aged <=132 years) holding a Ph.D. exhibit the highest spending behavior.
# Seniors (<=88.8 years) with graduate degrees closely follow, indicating a substantial spending pattern.
# In contrast, individuals with undergraduate qualifications, irrespective of age, tend to have the lowest spending behavior.
# These spending trends across different age and education segments highlight the varied consumer behaviors within the demographic. Understanding these patterns is crucial for tailoring marketing strategies and product offerings to better align with the preferences and financial capacities of specific customer segments.
# 
# 
# 
# 
# 
# 
# 

# In[66]:


# ANALYSING RELATIONSHIP BETWEEN INCOME ?EDUCATION AND CUSTOMER AGE GROUP
plt.figure(figsize=(10,8))
sns.barplot(x="Education",data=data1,y="Income",hue='Customer_agegroup')
plt.show()


# ## Observation
# ### Income Disparities Across Age Groups and Education Levels:
#     *Super centenarians (<=132 years) with Doctorate Degrees exhibit the highest income levels.
#     *Seniors (<=88 years) holding PhD, Graduate, and Postgraduate degrees also demonstrate substantial incomes.
#     *Notably, adults (<=45.6 years) across all education groups consistently have the lowest income levels.
#     These observations highlight the influence of both age and education on income. The data suggests that individuals with higher educational qualifications tend to have higher incomes, and this trend is particularly pronounced among super centenarians and seniors. Understanding these patterns can be crucial for tailoring targeted strategies and services based on the demographics of different age and education segments.

# In[67]:


data1.columns


# In[68]:


#ANALYSING NUMBER OF STORE PURCHASE BY NEW WEB PURCHASE
plt.figure(figsize=(18,8))
sns.scatterplot(x='NumStorePurchases',  data=data1,  y='NumWebPurchases',  hue="Income")
plt.show()


# # Observation
# From this plot we can see that most income was generated from store purchases.

# In[69]:


#ANALYSING CORRELATION BETWEEN THESE FEATURES

selected_columns = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',"Income",'GoldProds', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth']
selected_data = data1[selected_columns]
corr_matrix = selected_data.corr()
print(corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,square=True,fmt=".2f")


# 
# ## Observation Report:
# 
# Upon analyzing the provided plot, the following observations are noted:
# 
# Accepted Campaigns Contribution:
# 
# AcceptedCmp1 contributed to 27% of the income and exhibited a positive relationship with NumWebPurchases (15%).
# 
# AcceptedCmp2 generated 8% of the income, positively correlating with NewWebPurchases (3%).
# 
# AcceptedCmp3 generated a mere 1% of the income, with a positive correlation observed with NumWebPurchases (4%).
# 
# AcceptedCmp4 made an 18% income contribution, showing a positive relation with NewWebPurchases (16%).
# 
# AcceptedCmp5 was the most impactful, generating 33% of the income, and demonstrating a positive correlation with NewWebPurchases (14%).
# 
# Notably, AcceptedCampaign 3 displayed the least performance among the campaigns.
# 
# ### GoldProds Impact:
# 
# GoldProds significantly contributed, generating 33% of the income.
# 
# GoldProds displayed a positive relationship with the number of store purchases (39%), number of catalog purchases (44%), and new web visits per month (41%).
# 
# ### NumDealsPurchases Influence:
# 
# NumDealsPurchases had a negative impact, resulting in a -8% contribution to income. This implies a negative correlation with the number of purchases made with a discount.
# 
# ### NumWebPurchases Contribution:
# 
# NumWebPurchases had a substantial impact, contributing 39% to the income.
# 
# ### NumCatalogPurchases Significance:
# 
# NumCatalogPurchases showed a strong contribution, generating 59% of the income.
# However, it exhibited a strong negative correlation (52%) with the Number of web visits per month.
# NumStorePurchases and Relations:
# 
# NumStorePurchases played a crucial role, contributing 53% to the income.
# It had a strong positive correlation (52%) with the number of catalog purchases and the number of web purchases.
# Conversely, NumStorePurchases exhibited a strong negative correlation (43%) with the Number of web visits per month.
# 
# ### NumWebVisitsMonth and Its Impact:
# 
# NumWebVisitsMonth displayed a strong negative correlation (55%) with income. This suggests that as the number of web visits per month increases, income tends to decrease.
# 
# These findings provide valuable insights into the contributions and relationships between various factors, aiding in the identification of key drivers and areas for improvement in the company's marketing and sales strategies.
# 
# 
# 
# 
# 

# In[70]:


#descriptive statistics
data1.describe()


# In[71]:


#describe categorical features
data1.describe(include=(bool,object))


# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'data1'
numerical_columns = data1.select_dtypes(include='number').columns

# Calculate the number of subplots needed
num_plots = len(numerical_columns)

# Set the size of the plot
plt.figure(figsize=(15, 5 * (num_plots // 3 + (num_plots % 3 > 0))))

# Iterate through numerical columns and plot boxplots
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(num_plots // 3 + (num_plots % 3 > 0), 3, i)
    sns.boxplot(x=data1[column])
    plt.title(f'Boxplot of {column}')

# Adjust layout
plt.tight_layout()
plt.show()



# In[73]:


data1.columns


# In[74]:


data1=data1[data1["Income"]<=160000]
data1=data1[data1["Wines"]<=1400]
data1=data1[data1["Fruits"]<=125]
data1=data1[data1["MeatProducts"]<=1500]
data1=data1[data1["FishProducts"]<=200]
data1=data1[data1["SweetProducts"]<=250]
data1=data1[data1["GoldProds"]<=250]
data1=data1[data1["NumDealsPurchases"]<=8]
data1=data1[data1["NumWebVisitsMonth"]<=12.5]
data1=data1[data1["NumCatalogPurchases"]<=10]
data1=data1[data1["NumWebPurchases"]<=20]
print(f"the total number of data points after removing outliers:{len(data1)}")











# In[75]:


#To plot some selected features
cat_var = ["Income","Recency","Age","Spent_Amt","Is_Parent"]
sns.pairplot(data1[cat_var],hue="Is_Parent")


# ### Observation
# The analysis reveals a distinct pattern among customers who are parents – a majority of them tend to earn a lower income (<=200,000) and exhibit lower spending compared to their non-parent counterparts. Furthermore, in terms of recency, parents are more prevalent in the "Number of days since the customer's last purchase" metric.
# 
# Additionally, the data indicates that younger parents tend to spend less than customers who are not parents. This observation underscores the influence of parenthood on spending behavior, with age playing a noteworthy role.
# 
# Moreover, a noteworthy finding emerges in the analysis, revealing a positive linear relationship between the customer's income and the amount spent. As income increases, there is a corresponding upward trend in the amount spent, highlighting the significance of income in shaping customer spending patterns. This insight can inform targeted marketing strategies, tailored to the distinct characteristics of parent and non-parent segments, as well as the influence of income on customer spending.

# In[76]:


#Dropping the outliers by setting a cap on Age and income.
#data1=data1[(data1["Age"]<90)]
#data1=data1[(data1["Income"]<600000)]
#print(f"the total number of data points after removing the outliers are:{len(data1)}")


# In[77]:


data1.columns


# In[ ]:





# ### **UNSUPERVISED MACHINE LEARNING**
#  Naturally in Machine Learning, We often implemement UnSupervised ML models as either; a part of the Feature Engineering Process,i.e to either help us obtain more insights during the EDA, or as a step to reduce the dimensions to improve the performance of any Supervised ML model.
# 
# For this class simulation/ Case study , we will use the pca for feature decomposition and kmeans as a clustering algorithm, please feel free to research and test implememnt 'other' decomposition algortihms listed above and other clustering algorithms explored above too, compare the performances and you can decide to stick with a combination that suits you
# 

# ##**Dimensionality Reduction**
#  More input features often make a predictive modelling task more challenging to model, more generally referred to as the curse of dimensionality. thus,Dimensionality reduction refers to techniques that reduce the number of input variables in a dataset.
# 
# Principal Componenet Analysis (PCA) is a technique for redcuing the dimensions of a large dataset, increasing the interpreatblilty and at the same minimizing information loss
# other exmaples of Dimensionality Reduction techniques include Self Organizing Maps (SOM), t-distributed Stochastic Neighbor Embedding (t-SNE) etc.

# In[ ]:





# In[80]:


#creating a copy of the data set and dropping off redundant features
data2=data1.copy()


# In[81]:


#importing from sklearn lib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder,MinMaxScaler






#encode cat_var
s = (data2.dtypes == 'object')
cat_var = list(s[s].index)

encoder =LabelEncoder()
for i in cat_var:
    data2[i]=encoder.fit_transform(data2[[i]])


# In[82]:


data2.info()


# In[83]:


#scale dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data2),columns=data2.columns)
scaled_data

#scaler = MinMaxScaler()



# In[ ]:





# In[ ]:





# In[84]:


#instantiate PCA to reduce dimension
pca = PCA(n_components = 3)
PCA_ds=pd.DataFrame(pca.fit_transform(scaled_data),columns=(["col1","col2","col3"]))
PCA_ds


# In[85]:


pca.explained_variance_ratio_


# In[86]:


#visualizing our new data dimensions
x = PCA_ds['col1']
y = PCA_ds["col2"]
z = PCA_ds["col3"]

fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x,y,z,marker = "o")
ax.set_title("3d Visualization of new dimensions")


# In[87]:


#eigen values of our pca object



# In[ ]:





# 
# ## **K-means clustering**
# 
# **K-means clustering** is a method of vector quantization, originally from signal processing, that aims to **partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean** ([Wiki](https://en.wikipedia.org/wiki/K-means_clustering)). This is a method of **unsupervised learning** that learns the commonalities between groups of data without any target or labeled variable.
# 
# K-means clustering algorithm spits the records in the data into a **pre-defined number of clusters**, where the data points within each cluster are close to each other. One difficulty of using k-means clustering for customer segmentation is the fact that you need to know the number of clusters beforehand. Luckily, the silhouette coefficient can help you.
# 
# **The silhouette coefficient** measures how close the data points are to their clusters compared to other clusters. The silhouette coefficient values range from -1 to 1, where the closer the values are to 1, the better they are.
# 
# Let's find the best number of clusters:
# ### Clustering
# - elbow method to determine the number of clusters to be made
# - clustering via kmeans clustering
# - examine our new cluster by plotting

# In[88]:


get_ipython().system('pip install scikit-learn yellowbrick')

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer



# In[89]:


# Instantiate KElbowVisualizer
elbow = KElbowVisualizer(KMeans())

elbow.fit(PCA_ds)


# In[ ]:





# In[103]:


#fitting Kmeans algo to our dataset
kmeans = KMeans(n_clusters=7)
y_means = kmeans.fit_predict(PCA_ds)


#adding the clusters to the dimensions dataframe
PCA_ds["clusters"] = y_means

data1["clusters"]= y_means


# In[104]:


#PCA_ds
data1


# In[105]:


#plotting new data points
cmap="Accent"
fig = plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x,y,z,c=data1['clusters'],marker="o",cmap=cmap)
ax.set_title("3d Visualization of our data by their clusters")


# In[106]:


#import silhouette  score from sklearn
from sklearn.metrics import silhouette_score

#calculate the silhouette_score
silhouette_score = silhouette_score(data2,data1["clusters"],metric="euclidean")
print(f"Silhouette Score:{silhouette_score:.4f}")



# In[94]:


data1.head()


# 
# 

# In[95]:


#distribution of clusters

sns.countplot(x='clusters',data=data1)


# In[111]:


sns.scatterplot(x="Spent_Amt",data=data1,y="Income",hue="clusters")


# observation:
# From the above analysis we can see that the Amount spent by the customers increases as their income increases.

# In[97]:


#to plot out the clusters by income
sns.boxplot(x="clusters",data=data1,y="Income")


# observation
# clusters 2 and cluster 1 are low income earners
# clusters 4 are a mixture of low income earners and average income earners
# clusters 0 and cluster 5 are average income earners
# clusters 3 and cluster 6 are high income earners

# In[114]:


# To plot out the clusters by Amount Spent
sns.boxplot(x="clusters",data=data1,y="Spent_Amt")


# ### observations
# - cluster 4  and cluster 6 are low spenders
# - cluster 5 and cluster 0 are high spenders
# - cluster 3 are average spenders
# - cluster 1 and 2 are low and average spenders

# Based on the observation of the income and amount spent by the customers from the two plots above, we can draw the following conclusions:
# 
# Cluster 6: High income earners but low income spenders.
# Cluster 5: Average income earners but high income spenders.
# Cluster 4: A mixture of low and average income earners, but they are the least income spenders.
# Cluster 3: High income earners but average income spenders.
# Clusters 1 and 2: Low income earners and low income spenders.
# Cluster 0: High income earners and high income spenders.
# 
# 
# 
# 
# 

# In[117]:


# Multivariate analysis of Education,Wines and Clusters

sns.scatterplot(x="Education",data=data1,y="Wines",hue="clusters")


#  customers who are graduates and PhD holders purchased the most wines

# In[123]:


# Multivariate analysis of Customer_For,Spent_Amt and Clusters
cluster_colors = ["red","Blue","Purple","Green","Black","orange","yellow"]
sns.scatterplot(x="Customer_For",data=data1,y="Spent_Amt",hue="clusters", palette = cluster_colors)


# Based on the data analysis, we can draw the following conclusions regarding the spending behavior of customers over different durations:
# 
# Customers who have been with the company from 1 year to 7 years tend to spend the most.
# 
# Clusters 4 and 6 consistently spend the least amount regardless of the duration of their customer relationship.
# 
# Cluster 1 typically spends less than $500, slightly above this amount.
# 
# Cluster 3 generally spends less than $1000, slightly above this threshold.
# 
# Cluster 0 and 5 tend to spend between $1000  and  $2000 and slightly above this amount.
# 
# 
# 
# 
# 

# In[125]:


# Multivariate analysis of Is_Parent,Spent_Amt and Cluster
cluster_colors = ["red","Blue","Purple","Green","Black","orange","yellow"]
sns.scatterplot(x="Is_Parent",data=data1,y="Spent_Amt",hue="clusters",palette=cluster_colors)


# Based on the analysis, we can infer the spending patterns among different clusters, considering whether they are parents or not:
# 
# Clusters 6, 5, and 0, which consist of non-parents, tend to spend the most. Specifically, Cluster 6, although not parents, are the lowest spenders, typically spending less than $500.
# 
# Clusters 5 and 0 spend approximately between $500 and $2500. On the other hand, Clusters 4, 2, 1, and 3, comprising parents, are among the lowest spenders.
# 
# Cluster 4 exhibits the lowest spending behavior, followed by Cluster 2.
# 
# Cluster 1 tends to spend slightly above $500.
# 
# Cluster 3 spends a bit more, ranging from slightly above $500 to around $2000.
# 
# 
# 
# 
# 
# 

# Based on the analysis, we can discern spending patterns among different clusters, taking into account whether they are parents or not:
# 
# 1. **Clusters 6, 5, and 0 (Non-Parents)**:
#    - These clusters exhibit the highest spending tendencies.
#    - Specifically, Cluster 6, despite not being parents, tends to be the leest spender, typically spending **less than $500**.
# 
# 2. **Clusters 5 and 0 (Non-Parents)**:
#    - Their spending falls within the range of **$500 to $2500**.
# 
# 3. **Clusters 4, 2, 1, and 3 (Parents)**:
#    - These clusters are among the lowest spenders.
# 
# 4. **Cluster Insights**:
#    - **Cluster 4**: Demonstrates the lowest spending behavior.
#    - **Cluster 2**: Follows closely as another low spender.
#    - **Cluster 1**: Tends to spend slightly above **$500**.
#    
#    - **Cluster 3**: Spends a bit more, ranging from slightly above  **$500**   to around     **$2000**.
# 

# In[127]:


# Multivariate analysis of Is_Parent,Spent_Amt and Cluster
cluster_colors = ["red","Blue","Purple","Green","Black","orange","yellow"]
sns.scatterplot(x="Customer_agegroup",data=data1,y="Income",hue="clusters",palette=cluster_colors)


# Based on the analysis, it is observed that the customer age group categorized as Adults (age <= 45.6 years) tends to generate the highest income across all clusters. Following this, the age group categorized as Old Adults (age <= 67.2 years) represents the second-highest income bracket.
# 
# Conversely, the customer age group classified as Seniors (age <= 88.8 years) demonstrates the lowest income generation among all age categories.
# 

# In[130]:


data1.columns


# In[133]:


#ANALYSING CORRELATION BETWEEN THESE FEATURES

selected_columns = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',"Income",'GoldProds', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth','Wines', 'Fruits', 'MeatProducts',
       'FishProducts', 'SweetProducts','Spent_Amt']
selected_data = data1[selected_columns]
corr_matrix = selected_data.corr()
print(corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,square=True,fmt=".2f")


# Observation Report:
# Upon analyzing the provided plot, the following observations are noted:
# 
# Accepted Campaigns Contribution:
# 
# AcceptedCmp1 contributed to 33% of the Amount Spent and exhibited a positive relationship with NumWebPurchases (16%).
# 
# AcceptedCmp2 generated 14% of the Amount Spent, positively correlating with NewWebPurchases (4%).
# 
# AcceptedCmp3 generated a mere 2% of the Amount Spent, with a positive correlation observed with NumWebPurchases (5%).
# 
# AcceptedCmp4 made an 26% Amount Spent contribution, showing a positive relation with NewWebPurchases (18%).
# 
# AcceptedCmp5 was the most impactful, generating 45% of the Amount Spent, and demonstrating a positive correlation with NewWebPurchases (15%).
# 
# Notably, AcceptedCampaign 3 displayed the least performance among the campaigns.
# 
# GoldProds Impact:
# GoldProds significantly contributed, generating 54% of the Amount Spent.
# 
# GoldProds displayed a positive relationship with the number of store purchases (42%), number of catalog purchases (48%), and new web visits per month (22%).
# 
# NumDealsPurchases Influence:
# NumDealsPurchases had a negative impact, resulting in a -6% contribution to Amount Spent. This implies a negative correlation with the number of purchases made with a discount.
# 
# NumWebPurchases Contribution:
# NumWebPurchases had a substantial impact, contributing 59% to the Amount Spent.
# 
# NumCatalogPurchases Significance:
# NumCatalogPurchases showed a strong contribution, generating 81% of the Amount Spent. However, it exhibited a strong negative correlation (52%) with the Number of web visits per month. 
# 
# NumStorePurchases and Relations:
# 
# NumStorePurchases played a crucial role, contributing 72% to the Amount Spent. It had a strong positive correlation (61%) with the number of catalog purchases and the number of web purchases. Conversely, NumStorePurchases exhibited a strong negative correlation (-45%) with the Number of web visits per month.
# 
# NumWebVisitsMonth and Its Impact:
# NumWebVisitsMonth displayed a strong negative correlation (-49%) with Amount Spent. This suggests that as the number of web visits per month increases, Amount Spent tends to decrease.
# 
# These findings provide valuable insights into the contributions and relationships between various factors, aiding in the identification of key drivers and areas for improvement in the company's marketing and sales strategies.
# 
# 

# In[ ]:




