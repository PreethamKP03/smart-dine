# SmartDine: A Context-Aware Group Restaurant Recommendation System Using Machine Learning

SmartDine is an end-to-end Machine Learning system designed to solve the **Group Dining Dilemma**. Instead of recommending restaurants to single users, SmartDine aggregates multi-member preferences (divergent cuisines, budgets, dietary constraints, and neighborhood preferences) and scores candidate restaurants using a trained supervised ML regression model with explainable rationales.

---

## 1. Dataset & Preprocessing

* **Raw Dataset**: Bangalore Zomato restaurant dataset (`data/raw/zomato.csv`) containing **51,717 raw listing records** across 17 attributes.
* **Cleaning & Deduplication**:
  * Repeated branch/daily scrape listings were consolidated by physical restaurant entity (`name` + `address`), yielding **12,499 unique restaurants**.
  * `rate` string values (`4.1/5`, `NEW`, `-`) were parsed into numeric floats. Missing values were median-imputed.
  * Price for two (`approx_cost(for two people)`) was cleaned of commas and converted to numeric values (`cost_for_two`).
  * Boolean flags (`online_order`, `book_table`) were binary-encoded.
  * Output saved to `data/processed/restaurants_cleaned.csv` and `data/processed/restaurant_features.csv`.

---

## 2. Group Preference Aggregation Methodology

SmartDine does not perform blind arithmetic averaging of preferences. Instead, it applies multi-criteria group decision-making:
* **Cuisine Preference**: Normalized frequency and importance-weighted overlap score across all group members.
* **Group Budget**: Importance-weighted budget target with asymmetric penalty for over-budget venues.
* **Dietary Protection**: Strict safety filtering for vegetarian and vegan members (preventing venues serving conflicting menus).
* **Consensus / Group Agreement**: Measures standard deviation and mode concentration across member preferences to capture group harmony.

---

## 3. Context-Aware Features

Each candidate restaurant is evaluated against the group across 15 structured features:
1. `cuisine_match`: Jaccard-style overlap of member preferences against restaurant menu.
2. `budget_match`: Ratio penalty function for cost for two against member budgets.
3. `dietary_compatibility`: Strict compatibility check for vegetarian, vegan, and non-vegetarian safety.
4. `restaurant_type_match`: Alignment with desired ambiance (Cafe, Casual Dining, Pub/Bar, Quick Bites).
5. `location_match`: Proximity score across members' preferred Bangalore neighborhoods.
6. `booking_match`: Table reservation requirement satisfaction.
7. `online_order_match`: Online delivery compatibility.
8. `group_agreement`: Internal consensus metric among group members.
9. `rating_score`: Normalized Zomato rating [0..1].
10. `popularity_score`: Log-transformed vote volume.
11. `cost_for_two`: Absolute cost.
12. `votes`: Raw review volume.
13. `online_order`: Availability flag.
14. `book_table`: Availability flag.
15. `group_size`: Total count of dining members.

---

## 4. Machine Learning Model Comparison

Four supervised regression algorithms were benchmarked on the test dataset (`test_ml.csv`):

| Algorithm | MAE | RMSE | $R^2$ Score | Training Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.02399** | **0.03007** | **0.7641** | **0.54s** | 🏆 **Champion Model** |
| Random Forest | 0.02501 | 0.03139 | 0.7429 | 89.42s | Benchmark |
| Gradient Boosting | 0.02525 | 0.03167 | 0.7383 | 151.90s | Benchmark |
| Decision Tree | 0.02665 | 0.03346 | 0.7079 | 3.45s | Benchmark |

> **Synthetic Ground-Truth Methodology Note**: As the raw Zomato dataset does not contain real multi-person group interaction feedback, a synthetic ground-truth suitability target was synthesized using multi-attribute utility theory for the training experiment. Synthetic labels are used strictly for offline supervised training and are not claimed to be real-world customer ratings.

---

## 5. Project Structure

```
SmartDine/
│
├── data/
│   ├── raw/
│   │   └── zomato.csv                 # Raw Bangalore Zomato dataset (574 MB)
│   └── processed/
│       ├── restaurants_cleaned.csv     # 12,499 consolidated restaurant entities
│       ├── restaurant_features.csv     # Pre-computed feature matrix
│       ├── group_members.csv           # 100 synthetic benchmark groups
│       ├── train_ml.csv                # Training split (80%)
│       └── test_ml.csv                 # Testing split (20%)
│
├── notebooks/
│   ├── exploratory_analysis.ipynb      # EDA on ratings, costs, cuisines & locations
│   ├── 01_dataset_understanding.ipynb
│   └── 02_feature_engineering.ipynb
│
├── src/
│   ├── preprocessing.py               # Data cleaning & entity consolidation
│   ├── feature_engineering.py          # Restaurant-level feature derivation
│   ├── group_preferences.py            # Group aggregation & consensus scoring
│   ├── train_model.py                  # Model comparison & training pipeline
│   ├── evaluation.py                   # Test metrics & diagnostic plots
│   ├── recommendation.py               # Top-N recommendation engine
│   └── serving.py                      # Sub-50ms FastAPI serving backend
│
├── models/
│   ├── smartdine_best_model.joblib     # Serialized champion model
│   └── feature_columns.txt             # 15 input feature definitions
│
├── outputs/                            # Diagnostic plots, metrics & benchmark CSVs
├── frontend/                           # Modern luxury dark web dashboard
│   ├── index.html
│   ├── index.css
│   └── app.js
│
├── app.py                              # Streamlit Web Application
├── server.py                           # FastAPI / Web launcher
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. Running the Applications

### Option A: Streamlit Web Application
To run the interactive Streamlit UI:
```bash
streamlit run app.py
```
*Accessible at: `http://localhost:8501`*

### Option B: FastAPI Serving Dashboard
To run the ultra-fast real-time serving dashboard:
```bash
python server.py
```
*Accessible at: `http://localhost:8000`*
*API Documentation: `http://localhost:8000/docs`*
