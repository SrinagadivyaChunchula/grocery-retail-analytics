"""
Grocery Retail Analytics
========================
End-to-end analysis of grocery store sales data covering:
    - Discount & revenue impact analysis
    - Seasonal trends (monthly / quarterly)
    - City & region performance
    - Customer segmentation by age group
    - Payment method trends
    - Product & category performance
    - Store-level benchmarking

Dataset  : cleaned_grocery_data.csv  (10,000 transactions)
Author   : Srinaga Divya Chunchula
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── colour palette ──────────────────────────────────────────────────────────
PALETTE = "Set2"
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load raw CSV, enforce types, remove duplicates / nulls, flag high discounts."""
    df = pd.read_csv(path)
    print(f"[load]  Shape          : {df.shape}")
    print(f"[load]  Duplicate rows : {df.duplicated().sum()}")
    df = df.drop_duplicates()

    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"[load]  Missing values :\n{null_counts[null_counts > 0]}")
    df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    num_cols = ["Quantity", "Unit_Price_INR", "Discount_Percent", "Total_Amount_INR"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Business-rule filters
    df = df[(df["Quantity"] > 0) & (df["Unit_Price_INR"] > 0)]
    df = df[(df["Discount_Percent"] >= 0) & (df["Discount_Percent"] <= 80)]

    df["High_Discount_Flag"] = df["Discount_Percent"] >= 20
    print(f"[load]  Clean shape    : {df.shape}\n")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. OUTLIER DETECTION  (IQR method)
# ═══════════════════════════════════════════════════════════════════════════

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag outlier rows using IQR on Quantity, Unit_Price_INR, Total_Amount_INR."""
    cols = ["Quantity", "Unit_Price_INR", "Total_Amount_INR"]

    def _iqr_mask(series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

    for col in cols:
        df[f"{col}_outlier"] = _iqr_mask(df[col])

    df["any_outlier"] = df[[f"{c}_outlier" for c in cols]].any(axis=1)
    print("[outliers]  Per-column counts :")
    print(df[[f"{c}_outlier" for c in cols]].sum().to_string())
    print(f"[outliers]  Total outlier rows : {df['any_outlier'].sum()}\n")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Derive revenue metrics, time features, and mismatch validation flags."""
    df["Calculated_Total"] = (
        df["Unit_Price_INR"] * df["Quantity"] * (1 - df["Discount_Percent"] / 100)
    )
    df["Mismatch"]      = (df["Total_Amount_INR"] - df["Calculated_Total"]).abs()
    df["Mismatch_Flag"] = df["Mismatch"] >= 1

    df["Gross_Revenue"]   = df["Unit_Price_INR"] * df["Quantity"]
    df["Discount_Amount"] = df["Gross_Revenue"] - df["Total_Amount_INR"]

    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"]    = df["Date"].dt.isocalendar().week.astype(int)

    df["Revenue_Per_Unit"] = df["Total_Amount_INR"] / df["Quantity"]
    df["Is_Digital"]       = df["Payment_Method"].isin(["UPI", "Debit Card", "Credit Card"])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. ANALYSIS MODULES
# ═══════════════════════════════════════════════════════════════════════════

# ── 4a. Discount Analysis ───────────────────────────────────────────────────
def discount_analysis(df: pd.DataFrame):
    """Average discount, revenue loss, and discount-quantity correlation by category."""
    result = (
        df.groupby("Category")
        .agg(
            Avg_Discount=("Discount_Percent", "mean"),
            Total_Revenue=("Total_Amount_INR", "sum"),
            Total_Qty=("Quantity", "sum"),
            Transactions=("Transaction_ID", "count"),
        )
        .assign(
            Avg_Rev_Per_Txn=lambda x: x["Total_Revenue"] / x["Transactions"],
            Avg_Qty_Per_Txn=lambda x: x["Total_Qty"] / x["Transactions"],
            Revenue_Loss=df.groupby("Category")["Discount_Amount"].mean(),
        )
        .sort_values("Avg_Discount", ascending=False)
    )

    print("\n── Discount Analysis ──────────────────────────────────────")
    print(result[["Avg_Discount", "Revenue_Loss"]].round(2).to_string())
    corr = df["Discount_Percent"].corr(df["Quantity"])
    print(f"\nCorrelation (Discount vs Quantity): {corr:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    result["Avg_Discount"].plot(kind="bar", ax=axes[0], color=sns.color_palette(PALETTE, len(result)))
    axes[0].set_title("Average Discount % by Category")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Discount %")
    axes[0].tick_params(axis="x", rotation=45)

    result["Revenue_Loss"].sort_values(ascending=False).plot(
        kind="bar", ax=axes[1], color=sns.color_palette("flare", len(result))
    )
    axes[1].set_title("Avg Revenue Loss by Category")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Revenue Loss (INR)")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("discount_analysis.png")
    plt.close()
    print("[saved]  discount_analysis.png")


# ── 4b. Seasonal Analysis ──────────────────────────────────────────────────
def seasonal_analysis(df: pd.DataFrame):
    """Monthly & quarterly trends; per-category seasonality index."""
    monthly  = df.groupby("Month")["Total_Amount_INR"].sum()
    quarterly = df.groupby("Quarter")["Total_Amount_INR"].sum()

    print("\n── Seasonal Analysis ──────────────────────────────────────")
    print("Monthly Sales (top→bottom):")
    print(monthly.sort_values(ascending=False).to_string())
    print(f"\nBest Month  : {monthly.idxmax()}")
    print(f"Best Quarter: {quarterly.idxmax()}")

    pivot = (
        df.groupby(["Category", "Month"])["Total_Amount_INR"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot["Mean"] = pivot.mean(axis=1)
    pivot["Std"]  = pivot.std(axis=1)
    pivot["Seasonality_Index"] = pivot["Std"] / pivot["Mean"]
    si = pivot[["Seasonality_Index"]].sort_values("Seasonality_Index")
    print("\nCategory Seasonality (low = stable):")
    print(si.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    monthly.plot(kind="line", marker="o", ax=axes[0], color="steelblue")
    axes[0].set_title("Monthly Revenue Trend")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Revenue (INR)")
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    si["Seasonality_Index"].sort_values().plot(
        kind="barh", ax=axes[1], color=sns.color_palette(PALETTE, len(si))
    )
    axes[1].set_title("Seasonality Index by Category")
    axes[1].set_xlabel("Seasonality Index")
    plt.tight_layout()
    plt.savefig("seasonal_analysis.png")
    plt.close()
    print("[saved]  seasonal_analysis.png")


# ── 4c. City & Region Performance ─────────────────────────────────────────
def city_and_region_performance(df: pd.DataFrame):
    """Revenue and transaction counts by city and region."""
    city_rev  = df.groupby("City")["Total_Amount_INR"].sum().sort_values(ascending=False)
    region_rev = df.groupby("Region")["Total_Amount_INR"].sum().sort_values(ascending=False)

    print("\n── City & Region Performance ──────────────────────────────")
    print("City Revenue:")
    print(city_rev.to_string())
    print(f"\nTop City   : {city_rev.idxmax()}")
    print(f"Bottom City: {city_rev.idxmin()}")
    print("\nRegion Revenue:")
    print(region_rev.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    city_rev.plot(kind="bar", ax=axes[0], color=sns.color_palette(PALETTE, len(city_rev)))
    axes[0].set_title("Revenue by City")
    axes[0].set_ylabel("Revenue (INR)")
    axes[0].tick_params(axis="x", rotation=45)

    region_rev.plot(kind="bar", ax=axes[1], color=sns.color_palette("pastel", len(region_rev)))
    axes[1].set_title("Revenue by Region")
    axes[1].set_ylabel("Revenue (INR)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("city_region_performance.png")
    plt.close()
    print("[saved]  city_region_performance.png")


# ── 4d. Customer Analysis ──────────────────────────────────────────────────
def customer_analysis(df: pd.DataFrame):
    """Average spend, top categories, and UPI usage by customer age group."""
    avg_spend = df.groupby("Customer_Age_Group")["Total_Amount_INR"].mean().sort_values(ascending=False)

    print("\n── Customer Analysis ──────────────────────────────────────")
    print("Average Spend per Age Group:")
    print(avg_spend.round(2).to_string())

    cat_age = (
        df.groupby(["Customer_Age_Group", "Category"])["Total_Amount_INR"]
        .sum()
        .reset_index()
        .sort_values(["Customer_Age_Group", "Total_Amount_INR"], ascending=[True, False])
    )
    top3 = cat_age.groupby("Customer_Age_Group").head(3)
    print("\nTop 3 Categories per Age Group:")
    for age, cats in top3.groupby("Customer_Age_Group")["Category"].apply(list).items():
        print(f"  {age}: {cats}")

    upi_pct = (
        df[df["Payment_Method"] == "UPI"].groupby("Customer_Age_Group").size()
        / df.groupby("Customer_Age_Group").size()
        * 100
    )
    print("\nUPI Usage % by Age Group:")
    print(upi_pct.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    avg_spend.plot(kind="bar", ax=axes[0], color=sns.color_palette(PALETTE, len(avg_spend)))
    axes[0].set_title("Avg Spend by Age Group")
    axes[0].set_ylabel("Avg Spend (INR)")
    axes[0].tick_params(axis="x", rotation=30)

    upi_pct.plot(kind="bar", ax=axes[1], color=sns.color_palette("coolwarm", len(upi_pct)))
    axes[1].set_title("UPI Usage % by Age Group")
    axes[1].set_ylabel("UPI %")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("customer_analysis.png")
    plt.close()
    print("[saved]  customer_analysis.png")


# ── 4e. Payment Analysis ───────────────────────────────────────────────────
def payment_analysis(df: pd.DataFrame):
    """Payment method share, UPI vs Cash trend, and digital adoption by city."""
    share = (df["Payment_Method"].value_counts() / len(df) * 100).round(2)

    print("\n── Payment Analysis ────────────────────────────────────────")
    print("Payment Method Share (%):")
    print(share.to_string())

    trend = (
        df[df["Payment_Method"].isin(["UPI", "Cash"])]
        .groupby(["Month", "Payment_Method"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nUPI vs Cash Monthly Trend:")
    print(trend.to_string())

    city_digital = (df.groupby("City")["Is_Digital"].mean() * 100).sort_values(ascending=False)
    print("\nDigital Payment % by City:")
    print(city_digital.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    share.plot(kind="pie", ax=axes[0], autopct="%1.1f%%", startangle=140,
               colors=sns.color_palette(PALETTE, len(share)))
    axes[0].set_title("Payment Method Share")
    axes[0].set_ylabel("")

    trend.plot(kind="line", marker="o", ax=axes[1])
    axes[1].set_title("UPI vs Cash Monthly Trend")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Transactions")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("payment_analysis.png")
    plt.close()
    print("[saved]  payment_analysis.png")


# ── 4f. Product & Category Analysis ───────────────────────────────────────
def product_analysis(df: pd.DataFrame):
    """Revenue share, cumulative Pareto, top / slow-moving products."""
    total_rev = df["Total_Amount_INR"].sum()
    rev_share = (
        (df.groupby("Category")["Total_Amount_INR"].sum() / total_rev * 100)
        .reset_index()
        .rename(columns={"Total_Amount_INR": "Revenue_Share_%"})
        .sort_values("Revenue_Share_%", ascending=False)
    )
    rev_share["Cumulative_%"] = rev_share["Revenue_Share_%"].cumsum()

    print("\n── Product & Category Analysis ────────────────────────────")
    print("Revenue Share with Cumulative %:")
    print(rev_share.to_string(index=False))

    top10 = df.groupby("Product_Name")["Total_Amount_INR"].sum().nlargest(10)
    slow10 = df.groupby("Product_Name")["Quantity"].sum().nsmallest(10)
    print("\nTop 10 Products by Revenue:")
    print(top10.to_string())
    print("\nSlow-moving Products (lowest qty):")
    print(slow10.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(rev_share["Category"], rev_share["Revenue_Share_%"],
                color=sns.color_palette(PALETTE, len(rev_share)))
    ax2 = axes[0].twinx()
    ax2.plot(rev_share["Category"], rev_share["Cumulative_%"], color="red", marker="o")
    ax2.set_ylabel("Cumulative %")
    axes[0].set_title("Revenue Share by Category (Pareto)")
    axes[0].set_ylabel("Revenue Share %")
    axes[0].tick_params(axis="x", rotation=45)

    top10.plot(kind="barh", ax=axes[1], color=sns.color_palette("Blues_r", 10))
    axes[1].set_title("Top 10 Products by Revenue")
    axes[1].set_xlabel("Revenue (INR)")
    plt.tight_layout()
    plt.savefig("product_analysis.png")
    plt.close()
    print("[saved]  product_analysis.png")


# ── 4g. Store Analysis ─────────────────────────────────────────────────────
def store_analysis(df: pd.DataFrame):
    """Top/bottom stores, average transaction value, and below-city-avg stores."""
    store_rev = df.groupby("Store_ID")["Total_Amount_INR"].sum().sort_values(ascending=False)
    avg_txn   = df.groupby("Store_ID")["Total_Amount_INR"].mean()
    city_avg  = df.groupby("City")["Total_Amount_INR"].mean()

    print("\n── Store Analysis ──────────────────────────────────────────")
    print("Top 10 Stores:")
    print(store_rev.head(10).to_string())
    print("\nBottom 10 Stores:")
    print(store_rev.tail(10).to_string())

    below_avg = (
        df[["Store_ID", "City"]].drop_duplicates()
        .merge(avg_txn.rename("Store_Avg"), on="Store_ID")
        .assign(City_Avg=lambda x: x["City"].map(city_avg))
    )
    below_avg = below_avg[below_avg["Store_Avg"] < below_avg["City_Avg"]]
    print(f"\nStores below city average ({len(below_avg)} stores):")
    print(below_avg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    store_rev.head(10).plot(kind="bar", ax=axes[0], color=sns.color_palette(PALETTE, 10))
    axes[0].set_title("Top 10 Stores by Revenue")
    axes[0].set_ylabel("Revenue (INR)")
    axes[0].tick_params(axis="x", rotation=45)

    city_avg.sort_values(ascending=False).plot(
        kind="bar", ax=axes[1], color=sns.color_palette("muted", len(city_avg))
    )
    axes[1].set_title("Avg Transaction Value by City")
    axes[1].set_ylabel("Avg Txn (INR)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("store_analysis.png")
    plt.close()
    print("[saved]  store_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_clean_data(df: pd.DataFrame, path: str = "cleaned_grocery_data.csv"):
    drop_cols = [
        "Calculated_Total", "Mismatch", "Mismatch_Flag",
        "Quantity_outlier", "Unit_Price_INR_outlier",
        "Total_Amount_INR_outlier", "any_outlier",
    ]
    df.drop(columns=drop_cols, errors="ignore").to_csv(path, index=False)
    print(f"\n[export]  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    DATA_PATH = "cleaned_grocery_data.csv"   # update path if needed

    df = load_and_clean_data(DATA_PATH)
    df = detect_outliers(df)
    df = feature_engineering(df)

    discount_analysis(df)
    seasonal_analysis(df)
    city_and_region_performance(df)
    customer_analysis(df)
    payment_analysis(df)
    product_analysis(df)
    store_analysis(df)

    export_clean_data(df)
    print("\n✅  All analysis complete.")


if __name__ == "__main__":
    main()
