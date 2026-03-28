## Causal Churn Analysis and Prescription System
This repository contains an end-to-end pipeline that integrates Machine Learning Classification, Causal Inference, and Prescriptive Analytics to manage customer churn. The project uses a Gradio interface to provide actionable insights for individual users.

Core Capabilities
Churn Prediction: Estimates the probability of a user churning based on engagement and transaction metrics using RandomForestClassifier.

Causal Discovery (DoWhy): Moves beyond correlation to determine if specific features actually cause changes in churn probability.

Prescriptive Optimization (EconML): Utilizes Double Machine Learning (DML) to simulate discount scenarios and recommend the optimal discount rate to minimize churn risk for a specific user.

Interactivity: A multi-tab Gradio dashboard for step-by-step analysis.

##Project Workflow

1. Data Processing
Target Definition: Churn is defined by a combination of zero transactions and low app/web session activity (10th percentile).

Feature Engineering: Calculates metrics like avg_items_per_transaction and discount_rate.

2. Analysis Steps
Step 1 - Prediction: Trains a model on user features to generate a churn_prob. It also calculates potential financial loss based on the user's average monthly transactions.

Step 2 - Causal Estimation: Uses the DoWhy framework to identify the causal effect of a chosen treatment feature on the churn outcome while controlling for common causes (confounders).

Step 3 - Discount Suggestion: Employs EconML's DML to calculate the Conditional Average Treatment Effect (CATE). It tests 10%, 20%, and 30% discount increases to find the "Best Discount" for the selected user.

## Requirements
bash:
pip install pandas numpy scikit-learn dowhy econml gradio

## Usage
bash:
DataSet = pd.read_csv('your_path/GA4_DF.csv')
bash:
python main.py





