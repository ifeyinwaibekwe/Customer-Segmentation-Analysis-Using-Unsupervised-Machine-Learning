# Customer-Segmentation-Analysis-Using-Unsupervised-Machine-Learning
This project utilizes unsupervised machine learning techniques to conduct customer segmentation analysis, leveraging data from People and Products datasets, with insights empowering businesses to customize marketing strategies and enhance overall customer engagement through refined product offerings.


![Screenshot 2024-03-18 170733](https://github.com/ifeyinwaibekwe/Customer-Segmentation-Analysis-Using-Unsupervised-Machine-Learning/assets/149434454/65d6de9e-6e75-4098-bcd6-c973630b28f6)
![Screenshot 2024-03-18 170838](https://github.com/ifeyinwaibekwe/Customer-Segmentation-Analysis-Using-Unsupervised-Machine-Learning/assets/149434454/f538b912-b65e-403d-9133-fde3c37728d5)
![Screenshot 2024-03-18 171022](https://github.com/ifeyinwaibekwe/Customer-Segmentation-Analysis-Using-Unsupervised-Machine-Learning/assets/149434454/d722bb7b-3ca1-40c9-9351-dde9be5a5896)

## Table of Contents
- [Project Overview](#project-overview)
- [Project Overview](#project-objective)
- [Data Sources](#data-sources)
- [Data Processing](#data-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Insights](#key-insights)
- [Conclusion](#conclusion)


  ## Project Overview
  Facing challenges in effectively targeting and engaging individual customers, Goldie’s Foods and Cold Store seeks to overcome this hurdle by employing a data-driven approach. I conducted a comprehensive customer personality analysis by leveraging machine learning techniques to gain insights into customer behaviors, interests, and preferences, enabling the store to enhance its targeting strategies and foster deeper customer engagement.

  ## Project Objective
   The goal of this project is to perform Exploratory Analysis and a Customer Personality Analysis, this will aid you segment customers based on their purchasing behavior and demographic information. We will use unsupervised machine learning techniques like Dimensionality reduction (PCA) and Clustering to identify groups of customers with similar behavior and characteristics. This information can be used to develop targeted marketing campaigns, personalized product recommendations, etc

## Data Sources
Data Source
The data for this project consists of two main datasets: People and Products. The People dataset contains information about customers, including their unique identifier, demographic details, enrollment date, and purchasing history. The Products dataset includes details on the amount spent by customers on various product categories over the past two years.


## Data Preprocessing
Prior to analysis, the data will undergo preprocessing steps to handle missing values, standardize feature(Minmaxscaler), and address any inconsistencies. Exploratory Data Analysis (EDA) techniques was employed to gain insights into the distribution of variables, identify correlations,outliers and detect anomalies in the data. Additionally, feature engineering techniques were applied to extract relevant information from the raw data.


## Evaluation Metrics
To assess the effectiveness of our customer segmentation model, i employed various metrics and techniques. Initially, i utilized dimensionality reduction using PCA to streamline the feature space and improve interpretability. Subsequently, i employed the elbow method, facilitated by the KElbowVisualizer, to identify the optimal number of clusters for K-means clustering. Additionally, i utilize the silhouette score and euclidean metric to evaluate the cohesion and separation of clusters, offering insights into their quality and distinctiveness. By integrating these evaluation methods, we ensure robust and insightful segmentation outcomes, enabling Goldie’s Foods and Cold Store to effectively target and engage with their diverse customer base.

## Key Insight
### Observation Report:
Upon analyzing the corelation between some selected features, the following observations are noted:

### Accepted Campaigns Contribution:

AcceptedCmp1 contributed to 33% of the Amount Spent and exhibited a positive relationship with NumWebPurchases (16%).

AcceptedCmp2 generated 14% of the Amount Spent, positively correlating with NewWebPurchases (4%).

AcceptedCmp3 generated a mere 2% of the Amount Spent, with a positive correlation observed with NumWebPurchases (5%).

AcceptedCmp4 made an 26% Amount Spent contribution, showing a positive relation with NewWebPurchases (18%).

AcceptedCmp5 was the most impactful, generating 45% of the Amount Spent, and demonstrating a positive correlation with NewWebPurchases (15%).

Notably, AcceptedCampaign 3 displayed the least performance among the campaigns.

GoldProds Impact:
GoldProds significantly contributed, generating 54% of the Amount Spent.

GoldProds displayed a positive relationship with the number of store purchases (42%), number of catalog purchases (48%), and new web visits per month (22%).

NumDealsPurchases Influence:
NumDealsPurchases had a negative impact, resulting in a -6% contribution to Amount Spent. This implies a negative correlation with the number of purchases made with a discount.

NumWebPurchases Contribution:
NumWebPurchases had a substantial impact, contributing 59% to the Amount Spent.

NumCatalogPurchases Significance:
NumCatalogPurchases showed a strong contribution, generating 81% of the Amount Spent. However, it exhibited a strong negative correlation (52%) with the Number of web visits per month. 

NumStorePurchases and Relations:

NumStorePurchases played a crucial role, contributing 72% to the Amount Spent. It had a strong positive correlation (61%) with the number of catalog purchases and the number of web purchases. Conversely, NumStorePurchases exhibited a strong negative correlation (-45%) with the Number of web visits per month.

NumWebVisitsMonth and Its Impact:
NumWebVisitsMonth displayed a strong negative correlation (-49%) with Amount Spent. This suggests that as the number of web visits per month increases, Amount Spent tends to decrease.



## Based on the analysis, we can discern spending patterns among different clusters, taking into account whether they are parents or not:

1. **Clusters 6, 5, and 0 (Non-Parents)**:
   - These clusters exhibit the highest spending tendencies.
   - Specifically, Cluster 6, despite not being parents, tends to be the leest spender, typically spending **less than $500**.

2. **Clusters 5 and 0 (Non-Parents)**:
   - Their spending falls within the range of **$500 to $2500**.

3. **Clusters 4, 2, 1, and 3 (Parents)**:
   - These clusters are among the lowest spenders.

4. **Cluster Insights**:
   - **Cluster 4**: Demonstrates the lowest spending behavior.
   - **Cluster 2**: Follows closely as another low spender.
   - **Cluster 1**: Tends to spend slightly above **$500**.
   
   - **Cluster 3**: Spends a bit more, ranging from slightly above  **$500**  to around   **$2000**.

### Based on the data analysis, we can draw the following conclusions regarding the spending behavior of customers over different durations:

Customers who have been with the company from 1 year to 7 years tend to spend the most.

Clusters 4 and 6 consistently spend the least amount regardless of the duration of their customer relationship.

Cluster 1 typically spends less than $500, slightly above this amount.

Cluster 3 generally spends less than $1000, slightly above this threshold.

Cluster 0 and 5 tend to spend between  1000 𝑎𝑛𝑑 2000 and slightly above this amount.

### Based on the observation of the income and amount spent by the customer, we can draw the following conclusions:

Cluster 6: High income earners but low income spenders. Cluster 5: Average income earners but high income spenders. Cluster 4: A mixture of low and average income earners, but they are the least income spenders. Cluster 3: High income earners but average income spenders. Clusters 1 and 2: Low income earners and low income spenders. Cluster 0: High income earners and high income spenders.

## Conclusion and Recommendation
These findings provide valuable insights into the contributions and relationships between various factors, aiding in the identification of key drivers and areas for improvement in the company's marketing and sales strategies.
From the analysis of customer spending patterns and income levels across different clusters, several key insights emerge. Clusters 6, 5, and 0, comprised mainly of non-parents, demonstrate the highest spending tendencies, with Cluster 6 being the least spender among them. Conversely, Clusters 4, 2, 1, and 3, predominantly consisting of parents, exhibit lower spending behavior. Notably, Cluster 4 emerges as the lowest spender, followed closely by Cluster 2. Additionally, Cluster 3 tends to spend moderately more than Clusters 1 and 2, ranging from slightly above $500 to around $2000.

Recommendation:
To optimize marketing strategies and enhance customer engagement, Goldie’s Foods and Cold Store should tailor their approaches based on the spending patterns observed across different customer clusters. Targeted campaigns should be devised to appeal to the preferences and behaviors of high-spending clusters, such as Clusters 5 and 0, while strategies for improving engagement with lower-spending clusters, like Cluster 4, should also be explored. Moreover, personalized product recommendations and incentives could be implemented to encourage increased spending among various customer segments.
