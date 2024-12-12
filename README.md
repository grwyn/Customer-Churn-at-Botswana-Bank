# Customer Churn at Botswana Bank

Dataset source : https://www.kaggle.com/datasets/sandiledesmondmfazi/bank-customer-churn  
Deployment link : https://huggingface.co/spaces/grwyn/M2

Name        : Gerwyn Zulqarnain

There are six models that will be used for training in this project, along with a brief explanation of each model:
<ol>
    <li>Logistic Regression:<br>A linear model used for binary classification that predicts the probability of a class (usually 0 or 1) by using a logistic function.</li>
    <li>Decision Trees:<br>A tree-like model where data is split based on features to make predictions. Each node represents a decision based on a feature, and the leaves represent the output class.</li> 
    <li>Random Forest:<br>An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting by averaging their predictions.</li>  
    <li>Support Vector Classifier (SVC):<br>A classification algorithm that finds the best boundary (hyperplane) between classes by maximizing the margin between the closest data points of different classes.</li>
    <li>K-Nearest Neighbors (KNN):<br>A simple algorithm that classifies data points based on the majority class of their nearest neighbors in the feature space.</li>
    <li>XGBoost:<br>A powerful and efficient gradient-boosting algorithm that builds an ensemble of weak decision trees in a sequential manner, improving model performance by minimizing prediction errors.</li>
</ol>

The goal of this project is to build a robust machine learning pipeline to analyze customer churn at a Botswana bank. By exploring multiple classification models, we aim to identify the best-performing model based on the F1 score, which balances both precision and recall, minimizing false negatives and false positives. This will help the bank accurately predict customers likely to churn and implement targeted business strategies. The ultimate objective is to improve customer retention by leveraging insights from the machine learning model to understand the key factors driving churn, enabling the bank to optimize its customer engagement strategies and reduce churn rates effectively. Each model's performance will be assessed not only for accuracy but also in terms of generalizability, ensuring that the final model can be effectively deployed in real-world scenarios for predictive analytics.

## Exploratory Data Analysis (EDA)

The purpose of this exploratory analysis is to support the development of a machine learning model that predicts customer churn at Botswana Bank. By examining the factors influencing the target column (0 = Not Churned, 1 = Churned), the analysis aims to uncover patterns and key reasons behind why customers decide to leave the bank. This includes identifying significant variables, such as account usage, transaction history, customer demographics, and engagement levels, that correlate strongly with churn behavior. From these insights, we can provide actionable business recommendations to reduce churn, enhance customer retention strategies, and improve overall customer satisfaction.

### Churn Flag Distribution
<img src="images/Churn_flag.png" width="500" />

To build a machine learning model, I analyzed the distribution of the target column. Based on the pie chart visualization, the number of customers who did not churn is greater than those who did. With this insight, I will decide whether balancing is necessary. If the model shows signs of overfitting, I will apply balancing techniques; however, if the model demonstrates a good fit, balancing will not be required.

### Customer Segements
<img src="images/Customer_segment.png" width="500" />

The analysis of customer segments at Botswana Bank reveals three distinct categories: SME (Small and Medium Enterprises), Corporate, and Retail. The distribution of customers across these segments is relatively even, suggesting that the bank serves all three segments equally. This balanced distribution presents an opportunity to enhance customer loyalty by tailoring strategies to the specific needs and preferences of each segment.

To achieve this, it is crucial to analyze the products each segment is most likely to subscribe to and identify any patterns in their product preferences. Additionally, understanding the nature and frequency of complaints from each segment can provide valuable insights into areas for service improvement. For example, by addressing common issues raised by SMEs, optimizing corporate banking solutions, or enhancing retail banking experiences, Botswana Bank can strengthen customer satisfaction and retention within each segment.

### Balance
<img src="images/balance.png" width="500" />  

The analysis of customer balances reveals a significant gap between the average balance of customers who churned and those who did not, with churned customers having an average balance approximately 100,000 higher. This disparity suggests that customers with lower balances are less engaged or less attached to the bank. A lower balance might indicate minimal interaction with the bank’s services, fewer financial dependencies, or a lack of incentives to maintain their relationship with the institution. Consequently, this reduced attachment makes it easier for these customers to close their accounts or switch to competitors.

### Number of Complaints
<img src="images/Number_complaints.png" width="500" />

The analysis of the number of complaints in the Botswana Bank dataset reveals that customers who churned filed an average of nearly 7 complaints, whereas customers who did not churn reported an average of approximately 4.8 complaints. This suggests that while churned customers are more likely to have raised complaints, it is noteworthy that even non-churned customers reported a significant number of issues. This indicates that some level of dissatisfaction is present among all customer groups, highlighting that it is normal for a corporation to face challenges in fully meeting customer expectations.

However, the higher average number of complaints among churned customers suggests a tipping point where unresolved issues or negative experiences might contribute significantly to churn. This insight emphasizes the need for proactive complaint resolution mechanisms and better customer service practices. Addressing these issues promptly could help mitigate churn risks and improve overall customer satisfaction and loyalty.

### Number of Products
<img src="images/Number_of_products.png" width="500" />  

The analysis highlights the average number of products subscribed to by customers at Botswana Bank. Customers who churned subscribed to an average of 2 products, while those who remained loyal had subscriptions to 3 or more products on average. The gap between these two groups is not particularly large, indicating that the products offered by Botswana Bank are generally appealing and valuable to customers.

However, this finding suggests that having fewer product subscriptions may correlate with a higher likelihood of churn, possibly due to less engagement with the bank’s services or a weaker sense of dependency on the bank. To understand the root causes of churn more thoroughly, further analysis should focus on specific customer reasons for leaving. These could include dissatisfaction with service quality, unmet expectations, or more competitive offerings from other institutions. Identifying these factors will help Botswana Bank refine its strategies to retain customers and increase product adoption rates.

### Churned Customer Reason
<img src="images/Churned_customer_reason.png" width="500" />  

The distribution of customer churn reasons at Botswana Bank includes the following key factors: balance account closure, service issues, better offers elsewhere, and relocation. Each of these reasons offers valuable insights into customer behavior and areas for improvement:

1. Balance Account Closure:
Customers closing their accounts due to balance-related issues may indicate dissatisfaction with their account management, such as high fees, low interest rates, or limited incentives to maintain a relationship with the bank. This suggests an opportunity to introduce personalized account packages, waive fees for loyal customers, or offer tiered benefits for maintaining certain balance thresholds.

2. Service Issues:
Service-related problems are a common reason for churn, emphasizing the importance of improving customer service quality. This includes addressing delays, enhancing communication, and resolving complaints promptly. Investing in staff training, implementing advanced CRM systems, and offering multiple channels for customer support (e.g., chat, email, in-branch) could help mitigate service-related churn.

3. Better Offers Elsewhere:
Customers switching to competitors due to better offers indicate a need for Botswana Bank to remain competitive in its offerings. This includes revisiting product pricing, interest rates, and benefits, as well as crafting targeted promotions and loyalty programs. Analyzing competitor strategies can help the bank position itself more effectively in the market.

4. Relocation:
Churn due to relocation is typically beyond the bank's control, as it often involves customers moving to areas where Botswana Bank does not operate. However, this could be an opportunity to expand digital banking services, allowing customers to continue using their accounts remotely. Additionally, partnering with other banks in new locations or offering seamless transfer options could reduce the impact of relocation-related churn.

By understanding these reasons, Botswana Bank can adopt a segmented approach to customer retention. Efforts should focus on improving service quality, enhancing product offerings, and staying competitive in the market while leveraging technology to address challenges like relocation. Implementing these strategies could help reduce churn rates and improve overall customer satisfaction.

## Conculusion

### Exploratory Data Analysis (EDA)

The exploratory data analysis of Botswana Bank's customer churn data has provided valuable insights into the factors influencing customer retention and churn. Key findings include:

1. Churn Distribution:
Customers who did not churn are the majority, but churned customers represent a significant portion, requiring attention to improve retention strategies.

2. Customer Segments:
The SME, Corporate, and Retail segments are evenly distributed, highlighting the bank's balanced approach to serving various customer types. Tailored strategies for each segment are essential for enhancing satisfaction and loyalty.

3. Balance Analysis:
Churned customers tend to have higher average balances, suggesting that customers with lower balances are less engaged or attached to the bank.

4. Number of Complaints:
Churned customers report more complaints on average (7 vs. 4.8 for non-churned customers). This underscores the importance of addressing customer dissatisfaction promptly to prevent churn.

5. Number of Products:
Customers with fewer product subscriptions (average of 2 for churned vs. 3+ for non-churned) are more likely to churn. This suggests that increasing product adoption can strengthen customer relationships.

6. Reasons for Churn:
Major churn reasons include balance account closure, service issues, better offers elsewhere, and relocation. Each of these reasons highlights opportunities for targeted interventions to improve retention.

### Model
| | Logistic Regresion | Decision Tree | Random Forest | SVC | K-Neighbors Classifier | XGBoost |
| --- | --- | --- | --- | --- | --- | --- |
| F1 Score - Train | 1.0 | 1.0 | 1.0 | 1.0 | 0.99 | 1.0 |
| F1 Score - Test | 1.0 | 1.0 | 1.0 | 1.0 | 0.99 | 1.0 |

Based on the evaluation metrics presented, all models demonstrate exceptional performance on both the training and test sets, with F1 scores of 1.0 across most models, except for K-Neighbors Classifier (KNC), which achieves a slightly lower F1 score of 0.99. Despite this slight difference, KNC may still be the preferred choice due to the following reasons:

1. Simplicity and Interpretability:
KNC is a relatively simple and interpretable model. Its predictions are based on the similarity of data points in the feature space, which makes it easier to understand how the model reaches its decisions compared to more complex algorithms like Random Forest or XGBoost.

2. Avoiding Overfitting:
The slight drop in KNC's F1 score (from 1.0 to 0.99) on the test set suggests it may generalize slightly better to unseen data compared to other models that achieved perfect scores. Perfect test scores can sometimes indicate overfitting, particularly if the dataset is not very large or diverse.

3. Efficiency for Small Datasets:
KNC is computationally efficient for smaller datasets during training, as it does not build a model in the conventional sense but instead uses the dataset itself to make predictions. This could be an advantage depending on the size and complexity of the data.

4. Avoiding Complexity Overhead:
While more complex models like Random Forest or XGBoost may provide robust performance, they often come with higher computational costs and complexity in tuning hyperparameters. For this use case, if KNC delivers similar performance with less complexity, it becomes a more practical choice.

5. Flexibility in Distance Metrics:
KNC allows for flexibility in choosing distance metrics (e.g., Euclidean, Manhattan), which can be customized based on the nature of the data. This adaptability can enhance its performance when working with certain types of features.


| | Before Tuning | After Tuning |
| --- | --- | --- |
| F1 Score - Train | 0.9 | 1.0 |
| F1 Score - Test | 0.9 | 1.0 |

After tuning the K-Neighbors Classifier (KNC), the model's performance improved significantly, with both the training and test F1 scores rising from 0.9 to 1.0. This indicates that the hyperparameter tuning successfully optimized the model, enhancing its ability to capture patterns and generalize well to unseen data without overfitting. The perfect F1 scores suggest that the tuned model can reliably distinguish between churned and non-churned customers. This improvement underscores the importance of tuning in refining model performance, making KNC a strong candidate for deployment. However, further validation and monitoring for scalability are recommended as the dataset grows.

### Reccomendations

1. Enhance Customer Engagement:

- Develop personalized financial products and incentives for customers with lower balances to encourage engagement and long-term relationships.
- Create programs to educate customers about the benefits of utilizing multiple bank products.

2. Improve Customer Service:

- Implement robust complaint management systems to address customer issues swiftly and effectively.
- Invest in staff training and provide multiple communication channels (e.g., live chat, email, phone).

3. Retain High-Value Customers:

- Offer loyalty rewards or exclusive benefits for customers with higher balances to reinforce their relationship with the bank.

4. Competitive Analysis and Offers:

- Monitor competitor offerings to ensure that Botswana Bank remains competitive in terms of interest rates, fees, and product features.
- Launch targeted marketing campaigns and promotions to attract and retain customers.

5. Segment-Specific Strategies:

- Develop customized retention plans for SME, Corporate, and Retail segments based on their unique needs and preferences.
- Analyze product usage patterns within each segment to refine offerings and maximize value.

6. Expand Digital Services:

- Leverage digital banking solutions to retain customers impacted by relocation by offering seamless access to their accounts and services.

7. Focus on Proactive Retention:

- Use predictive analytics to identify customers at risk of churning and intervene proactively with tailored retention offers or service improvements.

By addressing these recommendations, Botswana Bank can significantly reduce churn rates, enhance customer satisfaction, and strengthen its competitive position in the market.
 



