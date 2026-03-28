import dowhy
from dowhy import CausalModel

import econml
from econml.dr import DRLearner

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import gradio as gr
from econml.dml import LinearDML

import streamlit as st



DataSet = pd.read_csv('C:/Users/yasla/Desktop/GA4_DF.csv')
DataSet.head(5)

# Changing Churn to our standard-----------------------------------------------

DataSet['is_churned'] = (
    (DataSet['transactions'] == 0) |
    ((DataSet['app_sessions'] < DataSet['app_sessions'].quantile(0.10))&(DataSet['web_sessions'] < DataSet['web_sessions'].quantile(0.10)))
).astype(int)



# extra DataSet encoding for avoiding future errors.---------------------------

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
DataSet['geo.country'] = le.fit_transform(DataSet['geo.country'])

#DataSet['user_id'] = DataSet['user_id'].str.replace('user_', '', regex=False)


DataSet['geo.country'] = DataSet['geo.country'].astype('category').cat.codes

le = LabelEncoder()
DataSet['geo.country'] = le.fit_transform(DataSet['geo.country'])


# new columns------------------------------------------------------------------

DataSet['avg_items_per_transaction'] = DataSet['items_viewed'] / DataSet['transactions']

DataSet['discount_rate'] = DataSet['ecommerce.discount'] / DataSet['ecommerce.purchase_value']


DataSet['avg_items_per_transaction'].replace([np.inf, -np.inf], 0, inplace=True)

# Declaring predict_proba as churn_prob column---------------------------------

features = [
    'user_lifetime_value',
    'geo.country',
    'user_properties.save_payment_info',
    'user_properties.push_opt_in',
    'event_name.add_to_wishlist',
    'web_sessions',
    'app_sessions',
    'transactions',
    'items_viewed',
    'ecommerce.discount',
    'items_viewed_per_app_session',
    'transactions_app',
    'items_added_to_cart_per_session_count',
    'user_engagement_proxy',
    'is_churned'
    
]
x = DataSet[features]
y = DataSet['is_churned']


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)


DataSet['churn_prob'] = model.predict_proba(x)[:, 1]

#The colon (:) before the comma means:
#"Take all rows."

#The 1 after the comma means:
#"Take column index 1 (second column)."

Data2 = DataSet.copy()

features = [
    'user_lifetime_value',
    'geo.country',
    'user_properties.save_payment_info',
    'user_properties.push_opt_in',
    'event_name.add_to_wishlist',
    'web_sessions',
    'app_sessions',
    'transactions',
    'items_viewed',
    'ecommerce.discount',
    'items_viewed_per_app_session',
    'transactions_app',
    'items_added_to_cart_per_session_count',
    'user_engagement_proxy',
    'is_churned'
    
]
x = Data2[features]
y = Data2['is_churned']


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)


Data2['churn_prob'] = model.predict_proba(x)[:, 1]


#------------------------------------------------------------------------------


# Global state to share between functions
global_state = {
    "DataSet": None,
    "user_id": None,
    "churn_prob": None
}

def User_Based_advanced_churn_predictor(input_str, user_id, threshold=0.6, tolerance=0.01):
    DataSet = Data2

    DataSet['user_lifetime_months'] = DataSet['user_lifetime_value'] / 30
    DataSet['user_lifetime_months'] = DataSet['user_lifetime_months'].replace(0, np.nan)
    DataSet['avg_monthly_transactions'] = DataSet['transactions'] / DataSet['user_lifetime_months']
    DataSet['avg_monthly_transactions'] = DataSet['avg_monthly_transactions'].fillna(0)

    A = [x.strip() for x in input_str.split(",")]
    x = pd.get_dummies(DataSet[A])
    y = DataSet['is_churned']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:, 1]
    report = classification_report(y_test, y_pred)

    importances = model.feature_importances_
    importance_info = "\nFeature Importances:\n"
    for feature, imp in zip(x.columns, importances):
        importance_info += f"- {feature}: importance = {imp:.4f}\n"

    high_risk_count = sum(y_probs > threshold)
    high_risk_info = f"\nModel detected that there are {high_risk_count} people with higher than {threshold} chance to churn.\n"

    churned_indices = x_test.index[y_pred == 1]
    lost_monthly_transactions = DataSet.loc[churned_indices, 'avg_monthly_transactions'].sum()
    loss_info = f"\nEstimated monthly lost transactions if churn happens: {lost_monthly_transactions:.2f} in total.\n"

    user_info = ""
    user_id = "user_" + user_id

    if user_id is not None:
        if user_id in DataSet['user_id'].values:
            user_row = DataSet[DataSet['user_id'] == user_id]
            user_x = pd.get_dummies(user_row[A])
            user_x = user_x.reindex(columns=x.columns, fill_value=0)
            user_prob = model.predict_proba(user_x)[0, 1]

            full_probs = pd.Series(model.predict_proba(x)[:, 1], index=x.index)
            similar_users_idx = full_probs[(full_probs >= user_prob - tolerance) & (full_probs <= user_prob + tolerance)].index
            similar_user_count = len(similar_users_idx)

            user_transaction_loss = user_row['avg_monthly_transactions'].values[0]
            similar_users_loss = DataSet.loc[similar_users_idx, 'avg_monthly_transactions'].sum()

            user_info = (
                f"\nUser ID: {user_id}\n"
                f"- Churn probability: {user_prob:.4f}\n"
                f"- Users with similar churn probability (tolerance of {tolerance}): {similar_user_count}\n"
                f"- Estimated monthly transaction loss if this user churns: {user_transaction_loss:.2f}\n"
                f"- Total estimated loss for similar users: {similar_users_loss:.2f}\n"
            )

            # Store for later DoWhy use
            global_state['DataSet'] = DataSet.copy() 
            DataSet['discount_rate'] = DataSet['ecommerce.discount'] / DataSet['ecommerce.purchase_value']
            DataSet['discount_rate'] = DataSet['discount_rate'].fillna(0)

            global_state['user_id'] = user_id
            global_state['churn_prob'] = full_probs.rename("churn_prob")

        else:
            user_info = f"\nUser ID {user_id} not found in dataset.\n"

    return  importance_info + high_risk_info + loss_info + user_info
 
def run_dowhy_on_feature(treatment_feature):
    DataSet = global_state['DataSet']
    user_id = global_state['user_id']
    churn_prob = global_state['churn_prob']

    if DataSet is None or user_id is None or churn_prob is None:
        return "Please run the churn predictor first."

    DataSet = DataSet.copy()
    DataSet['churn_prob'] = churn_prob

    common_causes = ['session_duration', 'promotion_clicks', 'transactions','items_viewed']  
    model = CausalModel(
        data=DataSet,
        treatment=treatment_feature,
        outcome="churn_prob",
        common_causes=common_causes
    )
    
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    effect_value = estimate.value
    if effect_value is None:
        return f"Causal effect could not be estimated for '{treatment_feature}'."

    direction = "increase" if effect_value > 0 else "decrease"
    effect_str = f"{effect_value:.4f}"

    return (
        f"Causal effect of **{treatment_feature}** on churn probability is **{effect_str}**.\n"
        f"This means if you increase **{treatment_feature}**, the churn probability will likely {direction}.\n"
    )

from econml.dml import DML


def discount_with_econml():
    
    data = global_state["DataSet"]
    user_id = global_state["user_id"]
    
    
    if data is None or user_id is None:
        return "Please run churn prediction first."

    if "discount_rate" not in data.columns:
        return "The dataset must contain a 'discount_rate' column."

    Y = global_state["churn_prob"]  
    T = data["discount_rate"].values  
    X = data.drop(columns=["user_id", "is_churned", "discount_rate"])
    
    model = DML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        model_final=RandomForestRegressor(),
        discrete_treatment=False
    )
 
    
    try:
        model.fit(Y=Y, T=T, X=X)
    
    except Exception as e:
        return f"Model fit error: {str(e)}"
    
    try:
        user_row = data[data["user_id"] == user_id]
        if user_row.empty:
            return "User not found."
          
        user_features = user_row.drop(columns=["user_id", "is_churned", "discount_rate"])
        user_features = user_features[X.columns]

        discount_base = user_row["discount_rate"].values[0]
        options = [discount_base + 0.10, discount_base + 0.20, discount_base + 0.30]
        
        effects = []
        marg_effect = model.const_marginal_effect(user_features)
        if marg_effect.ndim > 1:
            marg_effect = marg_effect[0][0]
            
        for d in options:
            
            delta = d - discount_base
            effect = float(marg_effect * delta)  
            effects.append(effect)

        
        best_idx = np.argmin(effects)
        best_discount = options[best_idx] 

        output = ""
        for opt, eff in zip(options, effects):
            output += f"{int(opt*100)}% discount → estimated churn reduction: {-eff:.4f}\n"
        
        output += f"\nBest discount: {int(best_discount * 100)}%"
        return output

    except Exception as e:
        return f"Prediction error: {str(e)}"




demo = gr.Blocks()

with demo:
    gr.Markdown("## Churn Predictor with Causal Inference")
    with gr.Tab("Step 1: Predict User Churn"):
        feature_input = gr.Textbox(label="Feature columns (comma separated)")
        user_input = gr.Textbox(label="User ID (just number part)")
        churn_output = gr.Textbox(label="Model Report")
        predict_btn = gr.Button("Run Prediction")
        predict_btn.click(User_Based_advanced_churn_predictor, inputs=[feature_input, user_input], outputs=churn_output)

    with gr.Tab("Step 2: Explore Causal Effect"):
        treatment_input = gr.Textbox(label="Treatment Feature (except : session_duration', 'promotion_clicks', 'transactions','items_viewed)")
        dowhy_output = gr.Textbox(label="DoWhy Effect Output", lines=10)
        dowhy_btn = gr.Button("Run Causal Estimation")
        dowhy_btn.click(fn=run_dowhy_on_feature, inputs=[treatment_input], outputs=dowhy_output)
        
    with gr.Tab("Step 3: Suggest Best Discount"):
        discount_output = gr.Textbox(label="Best Discount Suggestion", lines=8)
        suggest_btn = gr.Button("Run Discount Suggestion")
        suggest_btn.click(fn= discount_with_econml,
                          outputs=discount_output)


    with gr.Tab("Reset"):
        clear_btn = gr.Button("Reset Session")
        confirm = gr.Textbox(label="Status")
        def reset_all():
            global_state["DataSet"] = None
            global_state["user_id"] = None
            global_state["churn_prob"] = None
            return "Session cleared."
        clear_btn.click(reset_all, outputs=confirm)

if __name__ == "__main__":
    demo.launch()









