# 🛒 Grocery Retail Analytics Dashboard

> End-to-end data analysis project on 10,000+ grocery store transactions across 8 cities and 4 regions in India.

---

## 📌 Project Overview

This project performs a comprehensive analysis of grocery retail sales data using **Python**, **Pandas**, **Matplotlib**, and **Seaborn**. It covers discount impact, seasonal trends, customer behaviour, payment patterns, product performance, and store benchmarking.

---

## 🗂️ Project Structure

```
grocery-retail-analytics/
│
├── grocery_analysis.py          # Main analysis script (all 7 modules)
├── cleaned_grocery_data.csv     # Cleaned dataset (10,000 transactions)
├── requirements.txt             # Python dependencies
│
├── discount_analysis.png        # Output chart
├── seasonal_analysis.png        # Output chart
├── city_region_performance.png  # Output chart
├── customer_analysis.png        # Output chart
├── payment_analysis.png         # Output chart
├── product_analysis.png         # Output chart
└── store_analysis.png           # Output chart
```

---

## 📊 Analysis Modules

| # | Module | Key Insight |
|---|--------|-------------|
| 1 | **Discount Analysis** | Avg discount % per category; revenue loss; discount-quantity correlation |
| 2 | **Seasonal Analysis** | Monthly & quarterly trends; seasonality index per category |
| 3 | **City & Region Performance** | Revenue ranking across 8 cities and 4 regions |
| 4 | **Customer Analysis** | Avg spend by age group; top categories; UPI adoption |
| 5 | **Payment Analysis** | Payment method share; UPI vs Cash trend; digital adoption by city |
| 6 | **Product Analysis** | Revenue share (Pareto); top 10 products; slow-moving SKUs |
| 7 | **Store Analysis** | Top/bottom stores; stores below city average transaction value |

---

## 🔧 Tech Stack

| Tool | Usage |
|------|-------|
| Python 3.10+ | Core language |
| Pandas | Data cleaning, wrangling, aggregation |
| NumPy | Numerical operations, IQR outlier detection |
| Matplotlib | All charts and visualisations |
| Seaborn | Colour palettes and styling |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/SrinagadivyaChunchula/grocery-retail-analytics.git
cd grocery-retail-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python grocery_analysis.py
```

All 7 charts will be saved as `.png` files in the same directory.

---

## 📁 Dataset

- **Records**: 10,000 transactions
- **Columns**: Transaction ID, Date, Customer ID, Age Group, City, Region, Store ID, Category, Product Name, Brand, Quantity, Unit Price (INR), Discount %, Total Amount (INR), Payment Method
- **Date Range**: 2023 – 2024
- **Cities**: Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata, Mumbai, Pune
- **Regions**: North, South, East, West

---

## 💡 Key Findings

- 🏆 **North region** leads with ₹2.2 Cr revenue
- 📉 High discounts (≥20%) show weak positive correlation with quantity sold
- 📅 **Q4** is the strongest quarter across most categories
- 💳 **UPI** is the dominant payment method; digital adoption highest in metro cities
- 👥 **55+ age group** records the highest average spend per transaction

---

## 👩‍💻 Author

**Srinaga Divya Chunchula**  
Data Analyst | Python · SQL · Power BI  
📧 srinagadivyac@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/sri-naga-divya-chunchula-955b56288) | [GitHub](https://github.com/SrinagadivyaChunchula)
