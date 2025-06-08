# Querying Data Blog Article
### Querying Data: A Comprehensive Guide with Pandas, SQL, PySpark, and Polars

Working as a Data Scientist or Data Engineer often involves querying data from various sources. There are many tools and libraries available to perform these tasks, each with its own strengths and weaknesses. Also, there are many different ways to achieve similar results, depending on the tool or library used. It's important to be familiar with these different methods to choose the best one for your specific use case.

This article provides a comprehensive guide on how to query data using different tools and libraries, including Pandas, SQL, PySpark, and Polars. Each section will cover the setup, data creation, and various querying techniques such as filtering, grouping, joining, window functions, ranking, and sorting. The output will be identical across all tools, but the transformations will be implemented using the specific syntax and features of each library. Therefore allowing you to compare the different approaches and understand the nuances of each method.


## Setup

Before we start querying data, we need to set up our environment. This includes importing the necessary libraries, creating sample data, and defining constants that will be used throughout the article. The following sections will guide you through this setup process. The code for this article is also available on GitHub: [querying-data](...).


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


## SQL


### Create

```py
# Creates SQLite database and tables
conn: sqlite.Connection = sqlite3.connect(":memory:")

df_sales_pd.to_sql("sales", conn, index=False, if_exists="replace")
df_product_pd.to_sql("product", conn, index=False, if_exists="replace")
df_customer_pd.to_sql("customer", conn, index=False, if_exists="replace")
```

```py
# Verify SQL Connection
print("SQL Data:")
display(pd.read_sql("SELECT * FROM sales LIMIT 5", conn))
display(pd.read_sql("SELECT * FROM product LIMIT 5", conn))
display(pd.read_sql("SELECT * FROM customer LIMIT 5", conn))
```


### 1. Filtering and Selecting

```py
# Filter sales for a specific category
electronics_sales_sql = """
    SELECT *
    FROM sales
WHERE category = 'Electronics'
"""
electronics_sales: pd.DataFrame = pd.read_sql(electronics_sales_sql, conn)
print(f"Number of Electronics Sales: {len(electronics_sales)}")
display(pd.read_sql(electronics_sales_sql + "LIMIT 5", conn))

# Filter for high value transactions (over $500)
high_value_sales_sql = """
    SELECT *
    FROM sales
    WHERE sales_amount > 500
"""
high_value_sales: pd.DataFrame = pd.read_sql(high_value_sales_sql, conn)
print(f"Number of high-value Sales: {len(high_value_sales)}")
display(pd.read_sql(high_value_sales_sql + "LIMIT 5", conn))

# Select specific columns
sales_summary_sql = """
    SELECT date, category, sales_amount
    FROM sales
"""
sales_summary: pd.DataFrame = pd.read_sql(sales_summary_sql, conn)
print("Selected columns in Sales:")
display(pd.read_sql(sales_summary_sql + "LIMIT 5", conn))
```


### 2. Grouping and Aggregation

```py
# Basic aggregation
sales_stats_sql = """
    SELECT
        SUM(sales_amount) AS sales_sum,
        AVG(sales_amount) AS sales_mean,
        MIN(sales_amount) AS sales_min,
        MAX(sales_amount) AS sales_max,
        COUNT(*) AS sales_count,
        SUM(quantity) AS quantity_sum,
        AVG(quantity) AS quantity_mean,
        MIN(quantity) AS quantity_min,
        MAX(quantity) AS quantity_max
    FROM sales
"""
print(f"Sales Statistics: {len(pd.read_sql(sales_stats_sql, conn))}")
display(pd.read_sql(sales_stats_sql, conn))

# Group by category and aggregate
category_sales_sql = """
    SELECT
        category,
        SUM(sales_amount) AS total_sales,
        AVG(sales_amount) AS average_sales,
        COUNT(*) AS transaction_count,
        SUM(quantity) AS total_quantity
    FROM sales
    GROUP BY category
"""
print(f"Category Sales Summary: {len(pd.read_sql(category_sales_sql, conn))}")
display(pd.read_sql(category_sales_sql + "LIMIT 5", conn))

# Plot the results
fig: go.Figure = px.bar(
    pd.read_sql(category_sales_sql, conn),
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
sales_with_product_sql = """
    SELECT s.*, p.product_name, p.price
    FROM sales s
    LEFT JOIN product p ON s.product_id = p.product_id
"""
print(f"Sales with Product Data: {len(pd.read_sql(sales_with_product_sql, conn))}")
display(pd.read_sql(sales_with_product_sql + "LIMIT 5", conn))

# Join with customer information to get a complete view
complete_sales_sql = """
    SELECT
        s.*,
        p.product_name,
        p.price,
        c.customer_name,
        c.city,
        c.state
    FROM sales s
    LEFT JOIN product p ON s.product_id = p.product_id
    LEFT JOIN customer c ON s.customer_id = c.customer_id
"""
print(f"Complete Sales Data: {len(pd.read_sql(complete_sales_sql, conn))}")
display(pd.read_sql(complete_sales_sql + "LIMIT 5", conn))

# Calculate revenue and price difference
revenue_comparison_sql = """
    SELECT
        s.sales_amount,
        p.price,
        s.quantity,
        (p.price * s.quantity) AS calculated_revenue,
        (s.sales_amount - (p.price * s.quantity)) AS price_difference
    FROM sales s
    LEFT JOIN product p ON s.product_id = p.product_id
"""
print(f"Revenue Comparison: {len(pd.read_sql(revenue_comparison_sql, conn))}")
display(pd.read_sql(revenue_comparison_sql + "LIMIT 5", conn))
```


### 4. Window Functions

```py
# Time-based window function
daily_sales_sql = """
    SELECT
        date,
        SUM(sales_amount) AS total_sales
    FROM sales
    GROUP BY date
    ORDER BY date
"""
print(f"Daily Sales Data: {len(pd.read_sql(daily_sales_sql, conn))}")
daily_sales: pd.DataFrame = pd.read_sql(daily_sales_sql + "LIMIT 5", conn)

# Window functions for lead and lag
window_sql = """
    SELECT
        date AS sale_date,
        SUM(sales_amount) AS sales_amount,
        SUM(sales_amount) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_avg,
        LAG(SUM(sales_amount)) OVER (ORDER BY date) AS previous_day_sales,
        LEAD(SUM(sales_amount)) OVER (ORDER BY date) AS next_day_sales,
        SUM(sales_amount) - LAG(SUM(sales_amount)) OVER (ORDER BY date) AS day_over_day_change,
    FROM sales
    GROUP BY date
    ORDER BY date
"""
window_df: pd.DataFrame = pd.read_sql(window_sql, conn)
print(f"Window Functions: {len(window_df)}")
display(pd.read_sql(window_sql + "LIMIT 5", conn))

# Plot time series with rolling average
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=window_df["sale_date"],
        y=window_df["7d_moving_avg"],
        mode="lines",
        name="7-Day Moving Average",
        line=dict(width=3),
    )
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
customer_spending_sql = """
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        c.city,
        SUM(s.sales_amount) AS total_spending,
        RANK() OVER (ORDER BY SUM(s.sales_amount) DESC) AS rank
    FROM sales s
    JOIN customer c ON s.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_name, c.segment, c.city
    ORDER BY rank
"""
print(f"Customer Spending: {len(pd.read_sql(customer_spending_sql, conn))}")
display(pd.read_sql(customer_spending_sql + "LIMIT 10", conn))

# Rank products by quantity sold
product_popularity_sql = """
    SELECT
        p.product_id,
        p.product_name,
        p.category,
        SUM(s.quantity) AS total_quantity,
        RANK() OVER (ORDER BY SUM(s.quantity) DESC) AS rank
    FROM sales s
    JOIN product p ON s.product_id = p.product_id
    GROUP BY p.product_id, p.product_name, p.category
    ORDER BY rank
"""
print(f"Product Popularity: {len(pd.read_sql(product_popularity_sql, conn))}")
display(pd.read_sql(product_popularity_sql + "LIMIT 10", conn))
```


## PySpark

```py

```


### Create

```py

```


### 1. Filtering and Selecting

```py

```


### 2. Grouping and Aggregation

```py

```


### 3. Joining

```py

```


### 4. Window Functions

```py

```


### 5. Ranking and Partitioning

```py

```


### 6. Sorting

```py

```


## Polars

```py

```


### Create

```py

```


### 1. Filtering and Selecting

```py

```


### 2. Grouping and Aggregation

```py

```


### 3. Joining

```py

```


### 4. Window Functions

```py

```


### 5. Ranking and Partitioning

```py

```


### 6. Sorting

```py

```

## Conclusion
