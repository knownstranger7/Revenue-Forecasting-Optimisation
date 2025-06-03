import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from prophet import Prophet

st.set_page_config(page_title="Revenue Forecast & Optimization", layout="wide")

st.title(" Revenue Forecasting & Optimization Dashboard")

# Load monthly revenue data
@st.cache_data
def load_data():
    df = pd.read_csv('data/monthly_revenue.csv', parse_dates=['InvoiceMonth'])
    return df

monthly_revenue = load_data()

# Load Prophet model
model = joblib.load('models/prophet_model.pkl')
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

# Forecast Plot
st.subheader("Forecasted vs Actual Revenue (Prophet)")
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(monthly_revenue['InvoiceMonth'], monthly_revenue['TotalPrice'], label='Actual')
ax1.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Revenue')
ax1.legend()
st.pyplot(fig1)

# Promotional Impact Analysis
st.subheader("💡 Promotional Impact on Revenue")
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.boxplot(x='PromoActive', y='TotalPrice', data=monthly_revenue, ax=ax2)
ax2.set_title('Revenue Distribution With and Without Promotions')
st.pyplot(fig2)

# Dynamic Pricing Simulation
st.subheader("Dynamic Pricing Simulation")
base_revenue = monthly_revenue['TotalPrice'].mean()
base_margin = 0.4
discount_rates = np.linspace(0, 0.3, 10)
simulated_revenue = [base_revenue * (1 - dr) for dr in discount_rates]
simulated_profit = [rev * (base_margin - dr) for rev, dr in zip(simulated_revenue, discount_rates)]

fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.plot(discount_rates, simulated_revenue, label='Revenue')
ax3.plot(discount_rates, simulated_profit, label='Profit')
ax3.set_title('Revenue & Profit vs Discount Rate')
ax3.set_xlabel('Discount Rate')
ax3.set_ylabel('Amount')
ax3.legend()
st.pyplot(fig3)

st.markdown("**Built with Python, Prophet and Streamlit.**")
