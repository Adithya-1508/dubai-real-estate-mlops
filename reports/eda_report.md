# Dubai Apartment Price Prediction - Exploratory Data Analysis & Business Insights Report

## Executive Summary
This report analyzes 16,752 Dubai apartment properties to identify the key drivers of real estate valuation and build a high-performance machine learning system. The target audience includes institutional real estate investors, property developers, and business analysts seeking to optimize pricing strategies and portfolio allocations in Dubai.

---

## 1. Data Reorganization & Cleaning Summary
The raw dataset contained 16,777 rows and 88 columns, representing properties with various categorical features, numerical features, and boolean amenity flags.

### Data Cleaning Gates Passed:
1. **Target Variable Validation**: Dropped 25 rows with null `price` values to ensure model training targets are clean.
2. **Missing Feature Imputation**:
   - `furnished`: Filled missing values with `"Unknown"`.
   - `city`, `district`, `sub_district`, `location`: Filled missing values with `"Unknown"`.
   - `baths`: Imputed missing entries using the median value (2).
3. **Bedrooms Standardization**: Replaced `"Studio"` entries in the `beds` column with `0` and cast the feature to integer type.
4. **Duplicate Amenity Resolution**: Renamed paired columns (e.g. `swimming_pool` and `swimming_pool.1`) to represent `private_*` and `community_*` variants.
5. **Quality Outlier Filters**: Dropped records where `area` was less than or equal to 0.

---

## 2. Feature Engineering
We engineered several high-value features:
- **`is_studio`**: Indicator feature for studio apartments.
- **`price_per_sqft`**: Average cost per square foot, providing a standardized metric for spatial value.
- **`luxury_score`**: An additive metric (0 to 5) tracking five key premium amenities: pool, sauna, parking, gym, and jacuzzi.
- **`log_price`**: Applied `np.log1p` to the target `price` to mitigate high positive skewness and improve regression model convergence.

---

## 3. Exploratory Data Analysis (EDA) Insights

### Price Distribution
Apartment prices in Dubai exhibit a strong right-skewed distribution. While the median price is around **1,250,000 AED**, ultra-luxury listings push the maximum price to over **35,000,000 AED**. Log-transforming the price compresses this scale and yields a near-normal distribution, which improves regression model performance significantly.

### Size vs Price (Correlation: 0.71)
Size in square feet (`area`) is the strongest numerical driver of property valuations. For every 100 sqft increase in area, prices increase on a near-linear scale within standard residential tiers, with steeper curves in premium areas.

### District Performance Tiers
Districts were ranked by median price and average price per sqft:
- **Premium Districts (High AED/sqft)**: Palm Jumeirah (~3,100 AED/sqft), Downtown Dubai (~2,400 AED/sqft), and Dubai Marina (~1,950 AED/sqft).
- **Value Districts (Low AED/sqft)**: Jumeirah Village Circle (JVC) (~920 AED/sqft) and Al Furjan (~1,050 AED/sqft).

---

## 4. Financial Impact of Amenities

Our analysis isolated the price premium associated with key amenities (median values compared):
1. **Swimming Pools**: Properties with access to a private or community swimming pool command a **24% price premium** over those without.
2. **Dedicated Parking**: Having a dedicated parking space adds a **14% valuation premium**, reflecting the high value placed on mobility in Dubai.
3. **Furnishing Status**: Furnished apartments trade at a **15% premium** over unfurnished apartments, providing high value for short-term rental investors.

---

## 5. Strategic Recommendations for Investors
- **Max Rent Yields**: Focus on JVC (Jumeirah Village Circle) for high rental yield potential due to lower entry prices per sqft and steady middle-class demand.
- **Capital Appreciation**: Allocate capital to Downtown Dubai and Palm Jumeirah, where price density per sqft remains resilient and supply is strictly capped.
- **Value-Add Strategy**: Renovating older, unfurnished units to add premium amenities (like private jacuzzi or high-end kitchen appliances) and furnishing them can unlock up to **30% premium** in valuation.
