from transforms.api import transform, Output, TransformOutput
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
NUM_CUSTOMERS = 500
NUM_TRANSACTIONS = 10000
FRAUD_RATE = 0.02

CITY_COORDS = {
    "Boston": (42.3601, -71.0589),
    "New York": (40.7128, -74.0060),
    "Chicago": (41.8781, -87.6298),
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "Houston": (29.7604, -95.3698),
    "Seattle": (47.6062, -122.3321),
    "Miami": (25.7617, -80.1918),
}

MERCHANT_CATEGORIES = ["Grocery", "Electronics", "Restaurants", "Gas", "Travel", "Clothing", "Entertainment"]
DEVICE_TYPES = ["Mobile", "Web", "Tablet"]


def compute_risk_score(age, limit, city):
    base = 0
    if age < 25:
        base += 2
    elif age > 60:
        base += 1
    if limit >= 5000:
        base += 3
    elif limit >= 2500:
        base += 2
    else:
        base += 1
    high_risk_cities = ["New York", "Los Angeles", "Chicago", "San Francisco"]
    if city in high_risk_cities:
        base += 2
    else:
        base += 1
    if random.random() < 0.30:
        base += 2
    noise = random.randint(0, 2)
    return min(base + noise, 10)


def generate_customers():
    customers = []
    for cid in range(1, NUM_CUSTOMERS + 1):
        age = random.randint(18, 75)
        home_city = random.choice(list(CITY_COORDS.keys()))
        home_lat, home_lon = CITY_COORDS[home_city]
        monthly_limit = random.choice([500, 1000, 2500, 5000])
        risk = compute_risk_score(age, monthly_limit, home_city)
        customers.append({
            "customer_id": str(cid),
            "customer_age": age,
            "customer_home_city": home_city,
            "home_lat": home_lat,
            "home_lon": home_lon,
            "customer_monthly_limit": monthly_limit,
            "risk_score": risk,
        })
    return pd.DataFrame(customers)


def generate_transactions(customers_df):
    transactions = []
    for _ in range(NUM_TRANSACTIONS):
        customer = customers_df.sample(1).iloc[0]
        amount = round(random.uniform(5, customer.customer_monthly_limit * 1.5), 2)
        merchant = random.choice(MERCHANT_CATEGORIES)
        device = random.choice(DEVICE_TYPES)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 60))
        txn_city = random.choice(list(CITY_COORDS.keys()))
        txn_lat, txn_lon = CITY_COORDS[txn_city]
        txn_lat += random.uniform(-0.02, 0.02)
        txn_lon += random.uniform(-0.02, 0.02)
        is_fraud = 0
        if amount > 1.3 * customer.customer_monthly_limit:
            if random.random() < 0.5:
                is_fraud = 1
        if customer.risk_score >= 8 and txn_city != customer.customer_home_city:
            if random.random() < 0.25:
                is_fraud = 1
        if device == "Tablet" and random.random() < 0.05:
            is_fraud = 1
        if random.random() < FRAUD_RATE:
            is_fraud = 1
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": customer.customer_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "merchant_category": merchant,
            "device_type": device,
            "transaction_city": txn_city,
            "txn_lat": txn_lat,
            "txn_lon": txn_lon,
            "is_fraud": is_fraud,
        })
    return pd.DataFrame(transactions)


@transform(
    output_customers=Output("/palin-678c8d/Enterprise Fraud Detection(EFD)/Training Data/customers"),
    output_transactions=Output("/palin-678c8d/Enterprise Fraud Detection(EFD)/Training Data/transactions"),
)
def generate_training_data(output_customers: TransformOutput, output_transactions: TransformOutput):
    customers_df = generate_customers()
    transactions_df = generate_transactions(customers_df)
    output_customers.write_pandas(customers_df)
    output_transactions.write_pandas(transactions_df)
