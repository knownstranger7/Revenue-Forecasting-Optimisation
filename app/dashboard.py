import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from prophet import Prophet
from io import BytesIO
import base64

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

# === Function to display image with fixed width ===
def render_matplotlib(fig, width="700px"):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_html = f"<div style='text-align:center'><img src='data:image/png;base64,{base64.b64encode(buf.read()).decode()}' width='{width}'/></div>"
    st.markdown(img_html, unsafe_allow_html=True)

# === Forecast Plot ===
st.subheader("Forecasted vs Actual Revenue (Prophet)")
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(monthly_revenue['InvoiceMonth'], monthly_revenue['TotalPrice'], label='Actual')
ax1.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')

# Drop rows with NaNs before fill_between
forecast_clean = forecast.dropna(subset=['yhat_lower', 'yhat_upper'])

ax1.fill_between(
    forecast_clean['ds'],
    forecast_clean['yhat_lower'].astype(float),
    forecast_clean['yhat_upper'].astype(float),
    color='orange', alpha=0.2
)
ax1.set_xlabel('Date')
ax1.set_ylabel('Revenue')
ax1.legend()
render_matplotlib(fig1, width="720px")

# === Promotional Impact ===
st.subheader("💡 Promotional Impact on Revenue")
fig2, ax2 = plt.subplots(figsize=(7.5, 3.5))
sns.boxplot(x='PromoActive', y='TotalPrice', data=monthly_revenue, ax=ax2)
ax2.set_title('Revenue Distribution With and Without Promotions')
render_matplotlib(fig2, width="680px")

# === Dynamic Pricing Simulation ===
st.subheader("Dynamic Pricing Simulation")
base_revenue = monthly_revenue['TotalPrice'].mean()
base_margin = 0.4
discount_rates = np.linspace(0, 0.3, 10)
simulated_revenue = [base_revenue * (1 - dr) for dr in discount_rates]
simulated_profit = [rev * (base_margin - dr) for rev, dr in zip(simulated_revenue, discount_rates)]

fig3, ax3 = plt.subplots(figsize=(8.5, 3.8))
ax3.plot(discount_rates, simulated_revenue, label='Revenue')
ax3.plot(discount_rates, simulated_profit, label='Profit')
ax3.set_title('Revenue & Profit vs Discount Rate')
ax3.set_xlabel('Discount Rate')
ax3.set_ylabel('Amount')
ax3.legend()
render_matplotlib(fig3, width="720px")

st.markdown("**Built with Python, Prophet and Streamlit.**")
