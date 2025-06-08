# Querying Data Blog Article
### Querying Data: A Comprehensive Guide with Pandas, SQL, PySpark, and Polars


## Setup



### Imports

```py
# Import required libraries
import sqlite3
import numpy as np
import pandas as pd
import polars as pl
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession, functions as F
```

### Constants

```py
# Set seed for reproducibility
np.random.seed(42)
```

### Data

```py
# Generate sample data
n_records = 1000

# Create sales fact table
sales_data: dict[str, str] = {
    "date": pd.date_range(start="2023-01-01", periods=n_records, freq="D"),
    "customer_id": np.random.randint(1, 100, n_records),
    "product_id": np.random.randint(1, 50, n_records),
    "category": np.random.choice(
        ["Electronics", "Clothing", "Food", "Books", "Home"], n_records
    ),
    "sales_amount": np.random.uniform(10, 1000, n_records).round(2),
    "quantity": np.random.randint(1, 10, n_records),
}

# Create product dimension table
product_data: dict[str, str] = {
    "product_id": np.arange(1, 50),
    "product_name": [f"Product {i}" for i in range(1, 50)],
    "price": np.random.uniform(10, 500, 50).round(2),
    "category": np.random.choice(
        ["Electronics", "Clothing", "Food", "Books", "Home"], 50
    ),
    "supplier_id": np.random.randint(1, 10, 50),
}

# Create customer dimension table
customer_data: dict[str, str] = {
    "customer_id": np.arange(1, 100),
    "customer_name": [f"Customer {i}" for i in range(1, 100)],
    "city": np.random.choice(
        ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 100
    ),
    "state": np.random.choice(["NY", "CA", "IL", "TX", "AZ"], 100),
    "segment": np.random.choice(["Consumer", "Corporate", "Home Office"], 100),
}
```


## Pandas


### Create

```py
df_sales_pd = pd.DataFrame(sales_data)
df_product_pd = pd.DataFrame(product_data)
df_customer_pd = pd.DataFrame(customer_data)
```

```py
print("Pandas DataFrame:")
print(df_sales_pd.head(10))
print(df_product_pd.head(10))
print(df_customer_pd.head(10))
```


### 1. Filtering and Selecting

```py
# Filter sales data for specific category
electronics_sales: pd.DataFrame = df_sales_pd[df_sales_pd["category"] == "Electronics"]
print(f"Number of Electronics Sales: {len(electronics_sales)}")
display(electronics_sales.head())

# Filter for high value transactions (over $500)
high_value_sales: pd.DataFrame = df_sales_pd[df_sales_pd["sales_amount"] > 500]
print(f"Number of high-value Sales: {len(high_value_sales)}")
display(high_value_sales.head())

# Select specific columns
sales_summary: pd.DataFrame = df_sales_pd[["date", "category", "sales_amount"]]
display(sales_summary.head())
```


### 2. Grouping and Aggregation

```py
# Basic aggregation
sales_stats: pd.DataFrame = df_sales_pd.agg(
    {
        "sales_amount": ["sum", "mean", "min", "max", "count"],
        "quantity": ["sum", "mean", "min", "max"],
    }
)
display(sales_stats)

# Group by category and aggregate
category_sales: pd.DataFrame = df_sales_pd.groupby("category").agg(
    {
        "sales_amount": ["sum", "mean", "count"],
        "quantity": "sum",
    }
)
display(category_sales)

# Rename columns for clarity
category_sales.columns = [
    "total_sales",
    "average_sales",
    "transaction_count",
    "total_quantity",
]
display("Category Sales Summary:")

# Plot the results
fig: go.Figure = px.bar(
    category_sales.reset_index(),
    x="category",
    y="total_sales",
    title="Total Sales by Category",
    text="transaction_count",
    labels={"total_sales": "Total Sales ($)", "category": "Product Category"},
)
fig.show()
```


### 3. Joining

```py
# Join sales with product data
sales_with_product: pd.DataFrame = pd.merge(
    df_product_pd,
    df_product_id[["product_id", "product_name", "price"]],
    on="product_id",
    how="left",
)
display(sales_with_product.head())

# Join with customer information to get a complete view
complete_sales: pd.DataFrame = pd.merge(
    sales_with_product,
    df_customer_pd[["customer_id", "customer_name", "city", "state"]],
    on="customer_id",
    how="left",
)
display(complete_sales.head())

# Calculate revenue (price * quantity) and compare with sales amount
complete_sales["calculated_revenue"] = (
    complete_sales["price"] * complete_sales["quantity"]
)
complete_sales["price_difference"] = (
    complete_sales["sales_amount"] - complete_sales["calculated_revenue"]
)
display(
    complete_sales[
        ["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]
    ].head()
)
```


### 4. Window Functions

```py
# Time-based window function
df_sales_pd["date"] = pd.to_datetime(df_sales_pd["date"])  # Ensure date type
daily_sales: pd.DataFrame = (
    df_sales_pd.groupby(df_sales_pd["date"].dt.date).sum().reset_index()
)
daily_sales = daily_sales.sort_values("date")

# Calculate rolling averages (7-day moving average)
daily_sales["7d_moving_avg"] = (
    daily_sales["sales_amount"].rolling(window=7, min_periods=1).mean()
)

# Calculate lag and lead
daily_sales["previous_day_sales"] = daily_sales["sales_amount"].shift(1)
daily_sales["next_day_sales"] = daily_sales["sales_amount"].shift(-1)

# Calculate day-over-day change
daily_sales["day_over_day_change"] = (
    daily_sales["sales_amount"].pct_change() - daily_sales["previous_day_sales"]
)
daily_sales["pct_change"] = daily_sales["sales_amount"].pct_change() * 100

display(daily_sales.head())

# Plot time series with rolling average
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=daily_sales["date"],
        y=daily_sales["sales_amount"],
        mode="lines",
        name="Daily Sales",
    )
)
fig.add_trace(
    go.Scatter(
        x=daily_sales["date"],
        y=daily_sales["7d_moving_avg"],
        mode="lines",
        name="7-Day Moving Average",
    ),
    line=dict(width=3),
)
fig.update_layout(
    title="Daily Sales with 7-Day Moving Average",
    xaxis_title="Date",
    yaxis_title="Sales Amount ($)",
)
fig.show()
```


### 5. Ranking and Partitioning

```py
# Rank customers by total spending
customer_spending: pd.DataFrame = (
    df_sales_pd.groupby("customer_id")["sales_amount"].sum().reset_index()
)
customer_spending["rank"] = customer_spending["sales_amount"].rank(
    method="dense", ascending=False
)
customer_spending = customer_spending.sort_values("rank")

# Add customer details
top_customers: pd.DataFrame = pd.merge(
    customer_spending,
    df_customer_pd[["customer_id", "customer_name", "segment", "city"]],
    on="customer_id",
    how="left",
)
display(top_customers.head(10))

# Rank products by quantity sold
product_popularity: pd.DataFrame = (
    df_sales_pd.groupby("product_id")["quantity"].sum().reset_index()
)
product_popularity["rank"] = product_quantity["quantity"].rank(
    method="dense", ascending=False
)
product_popularity = product_quantity.sort_values("rank")

# Add product details
top_products: pd.DataFrame = pd.merge(
    product_popularity,
    df_product_pd[["product_id", "product_name", "category"]],
    on="product_id",
    how="left",
)
display(top_products.head(10))
```


### 6. Sorting

```py
# Sort sales data by date and sales amount
sorted_sales: pd.DataFrame = df_sales_pd.sort_values(
    by=["date", "sales_amount"], ascending=[True, False]
)
display(sorted_sales.head(10))

# Sort customers by total spending
sorted_customers: pd.DataFrame = top_customers.sort_values(
    by="sales_amount", ascending=False
)
display(sorted_customers.head(10))
```


## SQL


### Create


### 1. Filtering and Selecting


### 2. Grouping and Aggregation


### 3. Joining


### 4. Window Functions


### 5. Ranking and Partitioning


### 6. Sorting


## PySpark


### Create


### 1. Filtering and Selecting


### 2. Grouping and Aggregation


### 3. Joining


### 4. Window Functions


### 5. Ranking and Partitioning


### 6. Sorting


## Polars


### Create


### 1. Filtering and Selecting


### 2. Grouping and Aggregation


### 3. Joining


### 4. Window Functions


### 5. Ranking and Partitioning


### 6. Sorting
