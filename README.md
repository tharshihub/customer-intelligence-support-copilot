# Instamart Customer Intelligence & Support Copilot

## Overview

This project uses synthetic Instamart data to build a reproducible customer intelligence and support workflow. Part A audits, cleans, stores, and analyses customers, products, and orders. Parts B, C, and D cover segmentation and lapse prediction, ticket triage, and a policy-based refund agent. All monetary amounts are in **INR**. Only `completed` orders count toward revenue or spending unless explicitly stated otherwise.

## Reproduce

Run the following numbered commands in **Windows PowerShell**, from a clean clone. Replace the GitHub placeholders with the actual repository URL and folder name. Ensure the supplied `generate_data.py` and `mock_llm.py` are present and unmodified.

1. Clone and select `main`:
   ```powershell
   git clone https://github.com/tharshihub/customer-intelligence-support-copilot.git
   cd customer-intelligence-support-copilot
   git checkout main
   ```
2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
3. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Regenerate raw data:
   ```powershell
   python generate_data.py
   ```
6. Run the complete Parts A–D pipeline:
   ```powershell
   python run_project.py
   ```
7. Independently validate cleaned data:
   ```powershell
   python -m src.validate_data
   ```
   Expected successful output: `ALL CHECKPOINTS PASSED`.
8. Run SQL and pandas checks independently:
   ```powershell
   python -m src.sql_analysis
   python -m src.pandas_analysis
   ```

9. Run the Part D policy retrieval, assistant and evaluation:
   ```powershell
   python -m src.policy_retrieval
   python -m src.policy_assistant
   python -m src.policy_evaluation
   ```
10. Run the Part D refund tools and seven-scenario agent:
    ```powershell
    python -m src.refund_tools
    python -m src.refund_agent
    ```
11. Verify safety limits and checkpoint recovery:
    ```powershell
    python -m src.refund_safety_demo
    python -m src.refund_resume_demo
    ```
    Expected final messages: `D.Q6 ALL SEVEN SCENARIOS PASSED`,
    `D.Q7 ALL THREE SAFETY CHECKS PASSED`, and
    `D.Q8 CHECKPOINT AND RESUME PASSED`.

### Execution commands by assignment question

From the repository root, after activating `.venv` and installing `requirements.txt`, **`python run_project.py` executes the complete implemented pipeline for Parts A–D**. It invokes Part A audit, cleaning, validation, SQL and pandas analysis; all Part B modelling stages; Part C guardrails, redaction, prompts, comparison and routing; and Part D policy evaluation, ReAct scenarios, safety demonstrations and crash/resume. The table below provides **individual commands** to rerun a question without running the whole pipeline.

```powershell
# Generate input data first (not called by run_project.py)
python generate_data.py

# Run all implemented parts, A through D
python run_project.py
```

| Question | What it runs | Standalone command | In `run_project.py`? |
|---|---|---|---|
| A.Q1 | Generate synthetic inputs and reproduce pipeline | `python generate_data.py` then `python run_project.py` | Pipeline **yes**; data generation **no** |
| A.Q2 | Raw data audit | `python -m src.data_quality_audit` | Yes |
| A.Q3 | Customer cleaning | `python -m src.clean_customer_orders` | Yes, through cleaning pipeline |
| A.Q4 | Order cleaning | `python -m src.clean_customer_orders` | Yes, through cleaning pipeline |
| A.Q5 | Save and validate cleaned data | `python -m src.clean_customer_orders` then `python -m src.validate_data` | Yes, both |
| A.Q6 | Seven SQL analyses | `python -m src.sql_analysis` | Yes |
| A.Q7 | Pandas comparison and EDA | `python -m src.pandas_analysis` | Yes |
| B.Q1 | Customer RFM | `python -m src.customer_segmentation` | Yes |
| B.Q2 | K-means evaluation | `python -m src.customer_segmentation` | Yes |
| B.Q3 | Segment profiles and PCA | `python -m src.customer_segmentation` | Yes |
| B.Q4 | Historical lapse dataset and baseline | `python -m src.lapse_prediction` | Yes |
| B.Q5 | Lapse preprocessing | `python -m src.lapse_preprocessing` | Yes |
| B.Q6 | Model training and tuning | `python -m src.lapse_model_training` | Yes |
| B.Q7 | Class-balancing comparison | `python -m src.lapse_class_balancing` | Yes |
| B.Q8 | Interpretation, saved model and retention list | `python -m src.lapse_model_interpretation` | Yes |
| C.Q1 | Input guardrails | `python -m src.ticket_guardrails` | Yes |
| C.Q2 | PII redaction | `python -m src.ticket_redaction` | Yes |
| C.Q3 | Prompt definitions | `python -m src.ticket_prompts` | Yes |
| C.Q4 | Retry logic | `python -m src.ticket_llm_retry` | Indirectly through prompt comparison; standalone demonstration not separately called |
| C.Q5 | Response parsing and fallback | `python -m src.ticket_response_parser` | Indirectly through prompt comparison; standalone demonstration not separately called |
| C.Q6 | Zero-shot vs few-shot comparison | `python -m src.ticket_prompt_comparison` | Yes, with redacted tickets passed by runner |
| C.Q7(a) | Token-budget demonstration | `python -m src.ticket_triage` | Yes |
| C.Q7(b) | Final triage and routing | `python -m src.ticket_triage` | Yes |
| D.Q1 | Policy chunking | `python -m src.policy_retrieval` | Indirectly through policy evaluation; no separate call |
| D.Q2 | TF-IDF retrieval | `python -m src.policy_retrieval` | Indirectly through policy evaluation; no separate call |
| D.Q3 | Grounded policy answers | `python -m src.policy_assistant` | Indirectly through policy evaluation; no separate call |
| D.Q4 | Retrieval and answer evaluation | `python -m src.policy_evaluation` | Yes |
| D.Q5 | Refund tools and ledger | `python -m src.refund_tools` | Tools used by agent; standalone tool demonstration not separately called |
| D.Q6 | Seven ReAct scenarios | `python -m src.refund_agent` | Yes |
| D.Q7(a) | Maximum-iteration safeguard | `python -m src.refund_safety_demo` | Yes |
| D.Q7(b) | Token-limit safeguard | `python -m src.refund_safety_demo` | Yes |
| D.Q7(c) | Timeout safeguard | `python -m src.refund_safety_demo` | Yes |
| D.Q8 | Crash, checkpoint and resume | `python -m src.refund_resume_demo` | Yes |
| D.Q9 | ReAct vs plan-and-execute explanation | No command; see **D.Q9** below | Prints documentation pointer only |

**Execution notes:** Shared modules cover multiple questions (for example, B.Q1–B.Q3 and D.Q1–D.Q2); rerunning a shared module may execute the full module, not just one question. The standalone `src.ticket_prompt_comparison` entry point may construct its own inputs; `run_project.py` passes already-redacted tickets. The master runner calls `run_policy_evaluation()` rather than the separate policy-retrieval and policy-assistant demonstration entry points. `generate_data.py` must run before the pipeline when input files are absent. No plots are required or committed. Verify module entry points and dependencies against the final repository before submission. Regenerate `requirements.txt` from the final working environment with `python -m pip freeze > requirements.txt` before committing. `.gitignore` should include `.env`, `.venv/`, and `__pycache__/`.

## Part A

### Part A — Question-wise code map

| Question | File | Main function(s) | Purpose |
|---|---|---|---|
| A.Q1 | `generate_data.py`, `requirements.txt`, `run_project.py` | `main()` | Generate data and reproduce the pipeline. |
| A.Q2 | `src/data_quality_audit.py` | `run_audit()` | Audit raw shapes, types, missing values and anomalies. |
| A.Q3 | `src/clean_customer_orders.py` | `run_customer_cleaning()`, `clean_customers()` | Standardize customer city, gender, age and dates. |
| A.Q4 | `src/clean_customer_orders.py` | `run_order_cleaning()` | Deduplicate and clean amounts, timestamps and delivery times. |
| A.Q5 | `src/clean_customer_orders.py`, `src/validate_data.py` | `save_clean_data()`, `run_data_cleaning_pipeline()`, `validate_cleaned_data()` | Save clean CSV/SQLite tables and validate reference counts. |
| A.Q6 | `sql/queries.sql`, `src/sql_analysis.py` | `load_sql_queries()`, `run_sql_analysis()`, `validate_sql_results()` | Run seven SQL queries and check reference outcomes. |
| A.Q7 | `src/pandas_analysis.py` | `run_pandas_cross_check()`, `run_exploratory_analysis()`, `run_pandas_analysis()` | Compare pandas/SQL revenue and perform EDA. |

### A.Q1 — Reproducible project setup

`generate_data.py` creates the synthetic input datasets; `run_project.py` runs the completed Parts A–D pipeline using the implementations in `src/`. The environment setup and complete commands are documented under **Reproduce**, and individual question commands appear in **Execution Commands by Assignment Question**. The generated raw data files are stored in `data/raw/`.

### A.Q2 — Raw data audit

`src/data_quality_audit.py` reports each raw table's shape, data types, and the count and percentage of missing values **for every column**. The following values were verified in the supplied full execution log.

| Raw audit metric | Observed result |
|---|---:|
| Order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct `order_id` | 12,309 |
| Non-numeric `order_amount` | 120 |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` | 2,541 |
| Blank customer age | 61 |
| Age outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank gender spellings | 10 |
| Blank gender | 30 |

### A.Q3 — Customer cleaning rules and changed values

| Rule | Change and count |
|---|---|
| Normalize city | Strip whitespace and apply title case. **24 raw city spellings** reduce to **6 cities**. **Individual changed-cell count not printed in the log; measure separately if required.**. |
| Normalize gender | Map spellings to `male`, `female`, `unknown`. **10 nonblank raw spellings** consolidated; **30 blank genders** mapped to `unknown`. **Other changed-cell count not printed in the log; measure separately if required.**. |
| Validate age | **6** out-of-range raw ages set to missing. |
| Impute age | **61** raw blank ages plus **6** invalid ages require imputation, **67 raw values in total**, using city median with `groupby().transform('median')`. Final missing ages: **0**. |
| Parse signup date | Convert `signup_date` to datetime for all **1,200 customers**. **Final missing signup dates: 0; initial invalid-date count not printed.**. |

**Verified cleaned customer results:**

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

Gender counts: **female 610**, **male 560**, **unknown 30**. No missing ages; all cleaned ages between **18 and 60**.

### A.Q4 — Order cleaning rules and changed rows/values

The following rules execute **in this exact order**. Raw audit counts must not be confused with the number of values imputed **after deduplication**.

| Step | Rule | Count / result |
|---|---|---|
| 1 | Parse `order_ts` as datetime | Applied to 13,084 raw rows; invalid timestamps: **not separately reported in execution log**. |
| 2 | Remove exact duplicates | **119 rows removed**: 13,084 → 12,965. |
| 3 | Keep latest timestamp per `order_id` | **656 further rows removed**: 12,965 → 12,309 unique orders. |
| 4 | Convert `order_amount` to numeric; fill missing using `round(price_inr × quantity × (1 − discount_pct / 100), 2)` | **120 nonnumeric raw values**; **112** post-deduplication missing amounts filled. Final missing amounts: **0**. |
| 5a | Set negative `delivery_minutes` to missing | **106 raw negatives**; **105** post-deduplication negative times identified. Final negatives: **0**. |
| 5b | Impute missing delivery time for completed orders using completed-order category median with `groupby().transform('median')` | **2,541 raw blank delivery times**; **774** completed-order missing times filled. Final completed missing: **0**. |
| 5c | Keep cancelled delivery times missing | Final **1,199 cancelled orders** have missing delivery time; **newly cleared count not separately reported**. |

**Verified cleaned order results:** 12,309 rows; **11,110 completed**, **1,199 cancelled**, **0 pending**; no missing `order_amount`; no missing delivery time among completed orders; 1,199 missing delivery times among cancelled orders; completed revenue **INR 2,998,668.80**.

**Measured cleaning findings:** 119 exact duplicates and 656 additional repeated order IDs were removed; 112 post-deduplication missing amounts were filled; 105 negative delivery values were identified after deduplication; and 774 missing completed-order delivery times were imputed. The log reports 1,199 cancelled orders with missing delivery time after cleaning, but not how many were newly cleared. For full per-rule changed-cell reporting, instrument the cleaning functions for the remaining unreported counts.

### A.Q5 — Store and validate

Clean tables are saved to `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv` and written to SQLite database `data/instamart.db` as tables `customers`, `products`, and `orders`. Store `order_ts` as text formatted `YYYY-MM-DD HH:MM:SS` in SQLite. `src/validate_data.py` asserts the cleaning reference checkpoints and prints **`ALL CHECKPOINTS PASSED`** on success. The supplied execution log reports `ALL CHECKPOINTS PASSED`; an independent clean-clone reproduction remains a final submission check.

### A.Q6 — Seven SQL queries and reference outcomes

The seven queries are in `sql/queries.sql` and run through `pd.read_sql_query()` against `data/instamart.db`.

| Query | Question / technique | Verified result |
|---|---|---|
| Q1 | `WHERE` completed order amount > INR 500, then `HAVING` customer spend > INR 8,000 | **30 customers** |
| Q2 | Three-table join; completed revenue by city and category | **30 rows**, **INR 2,998,668.80** total; highest **Bengaluru / Electronics: INR 233,909.45** |
| Q3 | Scalar subquery; products above average catalogue price | **11 products** |
| Q4 | `LEFT JOIN`; customers without a completed order | **25 customers** |
| Q5 | Two CTEs; customers above average completed spend | **550 customers**, average **INR 2,552.06** |
| Q6 | City-level `RANK()`, `DENSE_RANK()`, `ROW_NUMBER()` | Six city leaders; **8 rows** where `RANK()` differs from `ROW_NUMBER()` |
| Q7 | Daily completed revenue and cumulative window sum | **183 days**, final cumulative **INR 2,998,668.80** |

**Q6 city leaders:**

| City | Customer | Completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

Tied spending values receive the same `RANK()`, whereas `ROW_NUMBER()` assigns unique sequential positions, resulting in eight rows with different values.

### A.Q7 — pandas cross-check and EDA

The completed-order city/category pivot uses `pd.pivot_table(..., margins=True)` and reproduces SQL Q2's **INR 2,998,668.80** total. City-level sum, mean, and count use named `groupby().agg()` aggregations.

| City | Sum | Mean | Count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |

| EDA statistic (completed orders) | Observed result |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | About 4.61 |
| Q1 | INR 64.80 |
| Q3 | INR 297.00 |
| IQR | INR 232.20 |
| Lower fence | INR −283.50 |
| Upper fence | INR 645.30 |
| Below lower fence | 0 orders |
| Above upper fence | 897 orders |

**Interpretation :**

**What is this data about?** The dataset contains synthetic Instamart customer purchases, product categories, discounts, quantities, order amounts, and delivery times. Revenue analyses use completed orders only.

**What do its distributions and statistics look like?** Completed order amounts are right-skewed: the mean is INR 269.91, median INR 135.00, and skewness approximately 4.61. Q1 is INR 64.80 and Q3 is INR 297.00.

**Are there outliers or anomalies?** The 1.5×IQR fences are INR −283.50 and INR 645.30. There are zero lower outliers and 897 upper outliers. Large orders may be valid purchases rather than errors.

**What relationships exist between variables?** The completed-order correlation matrix compares `order_amount`, `discount_pct`, `quantity`, and `delivery_minutes`. The strongest absolute correlation is between `order_amount` and `discount_pct` (about −0.095), versus +0.087 for `order_amount` and `quantity`. These relationships are very weak; correlation of this size does not establish causation.

**Part A validation:** The supplied run reports successful CSV and SQLite output creation, all seven SQL checkpoints passing, and pandas/SQL completed revenue agreeing at INR 2,998,668.80. The remaining submission checks are independent clean-clone reproducibility and precise changed-cell counts for any cleaning rules not fully instrumented.

## Part B

Customer Segments and Lapse Prediction. All eight question-level implementations passed the checks reported in the supplied execution log. The saved artifacts are `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, and `outputs/retention_list.csv`.

### Part B — Question-wise code map

| Question | File | Main function(s) | Purpose |
|---|---|---|---|
| B.Q1–B.Q3 | `src/customer_segmentation.py` | Module entry point (`python -m src.customer_segmentation`) | Compute RFM, compare K-means choices, profile and label segments, validate PCA, save segment CSV. |
| B.Q4 | `src/lapse_prediction.py` | `run_lapse_dataset()` | Define pre-cutoff cohort and September lapse label; construct historical features and baseline. |
| B.Q5 | `src/lapse_preprocessing.py` | `run_lapse_preprocessing()` | Fit training-only preprocessing and validate no missing features. |
| B.Q6 | `src/lapse_model_training.py` | `run_lapse_model_training()` | Tune Logistic Regression and Random Forest with five-fold CV; evaluate untouched test set. |
| B.Q7 | `src/lapse_class_balancing.py` | `run_class_balancing()` | Oversample only training lapsers and compare precision, recall and F1. |
| B.Q8 | `src/lapse_model_interpretation.py` | `run_model_interpretation()` | Explain features, save/reload fitted model, rank Premium test customers. |

**Execution note:** `run_model_interpretation()` calls B.Q6 training internally, and B.Q6 rebuilds the B.Q4 dataset; the master runner may therefore print some checks more than once. The B.Q1–B.Q3 module is executable as `python -m src.customer_segmentation`; do not assume a `run_customer_segmentation()` function exists.

### B.Q1 — Recency, Frequency, Monetary (RFM)

RFM is calculated from **completed orders** for customers with at least one completed order, using the **2025-10-01 00:00:00** snapshot. Recency is whole days since the most recent completed order; frequency is the number of distinct completed orders; monetary is total completed-order spend in INR.

| RFM checkpoint | Observed result |
|---|---:|
| Eligible customers | **1,175** |
| Median recency | **19 days** |
| Recency range | **0–180 days** |
| Mean frequency | **9.46** completed orders |
| Maximum frequency | **33** |
| Mean monetary | **INR 2,552.06** |
| Maximum monetary | **INR 14,766.55** |

**Finding:** Customers vary considerably in ordering activity and spend, making recency, frequency and monetary value useful inputs for actionable customer segmentation. **B.Q1 validation passed.**

### B.Q2 — K-means evaluation and K selection

RFM features were standardized with `StandardScaler`. K-means used `n_init=10` and `random_state=42`; candidate values **K=2 through K=8** were evaluated using inertia and silhouette score.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.761 | 0.465 |
| 3 | 1268.235 | 0.445 |
| **4** | **743.071** | **0.526** |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

**Chosen K = 4.** K=4 had the highest observed silhouette score (0.526) and a substantial inertia reduction relative to K=3, while producing interpretable business segments. K=8 has lower inertia than K=2 because additional clusters mechanically reduce within-cluster distances, but its lower silhouette (0.412 versus 0.465) and greater complexity do not make it the better business choice. **B.Q2 validation passed.**

### B.Q3 — Segment profiles, business names and PCA

| Segment ID | Name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Regular | 436 | 28.86 | 3.77 | 1,057.62 | Personalized recommendations and repeat-order offers |
| 1 | **Premium** | **138** | 24.12 | 6.77 | **7,129.70** | VIP service and exclusive retention offers |
| 2 | Frequent Loyal | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty rewards and referral incentives |
| 3 | At Risk | 182 | 109.04 | 2.20 | 685.99 | Targeted reactivation campaigns |

The **Premium** segment is the segment with the highest mean monetary spend (INR 7,129.70). The **Frequent Loyal** segment has the highest mean frequency (19.40), while **At Risk** has by far the highest mean recency (109.04 days).

PCA on standardized RFM features yielded **PC1 = 0.6343**, **PC2 = 0.2180**, and **total explained variance = 0.8523 (85.23%)**. The segment assignments and Premium flag were saved to `outputs/customer_segments.csv`. **B.Q3 validation passed.**

### B.Q4 — Historical lapse cohort, labels and baseline

The feature cutoff is **2025-09-01 00:00:00**. The cohort includes customers with a completed order before the cutoff; a customer is labelled **`lapsed=1`** if they had no completed order during **September 1–30, 2025**. Features are derived only from orders before the cutoff, together with customer attributes, and do **not** include B.Q3 whole-window segment labels.

The 11 features are `recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `favourite_category`, `city`, `age`, `gender`, and `tenure_days`.

| Cohort checkpoint | Observed result |
|---|---:|
| Historical cohort | **1,156 customers** |
| Lapsed in September | **443 customers** |
| Lapse rate | **38.32%** |
| Training rows | **924** |
| Test rows | **232** |
| Missing feature values | **0** |
| Majority-class baseline accuracy | **0.6164 (61.64%)** |
| Baseline lapsed precision / recall / F1 | **0 / 0 / 0** |

The stratified split used `test_size=0.2` and `random_state=42`. The baseline predicts all customers will not lapse; its accuracy is misleading for retention because it identifies **no actual lapsers**. **B.Q4 validation passed.**

### B.Q5 — Leakage-free preprocessing

A `ColumnTransformer` applies `StandardScaler` to eight numeric features and `OneHotEncoder(drop="first", handle_unknown="ignore")` to the three categorical features (`favourite_category`, `city`, `gender`). Preprocessing is fitted on training data only and is included inside each model's cross-validation pipeline.

| Preprocessing checkpoint | Observed result |
|---|---:|
| Original features | **11** |
| Transformed training matrix | **924 × 19** |
| Transformed test matrix | **232 × 19** |
| Scaler mean for training frequency | **8.1147** |
| Full-cohort frequency mean | **8.0285** |

The frequency means differ because the scaler is fitted to the training subset, while the full-cohort mean includes test customers. All 11 source features have no missing values, and fitted preprocessing does not use the test data. **B.Q5 validation passed.**

### B.Q6 — Model tuning, test evaluation and selection

Both models were tuned with **five-fold cross-validation**, scoring **lapsed-class F1**, using an unfitted preprocessing-and-model pipeline inside `GridSearchCV`.

| Model | Search space | Best parameters | Best training CV F1 |
|---|---|---|---:|
| Logistic Regression | `C=[0.01,0.1,1,10,100]` (25 CV fits) | **C=100** | **0.6506** |
| Random Forest | `max_depth=[3,5,8]`, `min_samples_leaf=[5,10,20]` (45 CV fits), 100 trees | **max_depth=8, min_samples_leaf=10** | **0.6866** |

**Untouched test-set results (`lapsed=1` positive):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.6164 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | **0.7284** | **0.6444** | **0.6517** | **0.6480** |
| **Random Forest (selected)** | 0.7198 | 0.6333 | 0.6404 | **0.6369** |

- Logistic Regression confusion matrix (actual rows, predicted columns): **`[[111,32],[31,58]]`**; FP=32, FN=31.
- Random Forest confusion matrix: **`[[110,33],[32,57]]`**; FP=33, FN=32.

**Selection rationale:** We selected **Random Forest** because it achieved the higher five-fold **training cross-validation F1 (0.6866 versus 0.6506)**, without using the test set to select the model. Logistic Regression subsequently achieved a slightly higher test F1 (0.6480 versus 0.6369), which is reported transparently rather than used to change the selection. A false negative misses a customer who will lapse and may lose future revenue; a false positive targets a non-lapsing customer and may waste retention incentives. We therefore assess both precision and recall, and would tune thresholds in future using the actual costs of missed customers and unnecessary outreach. The selected model's test F1 **exceeds the required 0.55**.

**Reference comparison:** The observed best Logistic Regression `C=100` differs from the assignment's illustrative `C≈1`; its test F1 of 0.6480 is nevertheless close to the reference ~0.62. Random Forest's best depth 8 and leaf size 10 match the reference, and its test F1 of 0.6369 is close to ~0.65. Differences in the Logistic Regression best C may reflect dataset or feature/preprocessing details; the execution log alone does not establish a specific cause. **B.Q6 validation passed.**

### B.Q7 — Training-only class balancing

The minority `lapsed=1` class was oversampled **only in the training set**, with replacement and `random_state=42`. All 924 original training rows were retained; 216 lapsed rows were duplicated to equalize the classes. The test set remained unchanged.

| Training class | Before | After |
|---|---:|---:|
| Not lapsed (0) | 570 | 570 |
| Lapsed (1) | 354 | 570 |
| **Total** | **924** | **1,140** |

The best Logistic Regression (`C=100`) was refitted and evaluated on the same 232 test customers:

| Metric | Original Logistic Regression | Oversampled Logistic Regression | Change |
|---|---:|---:|---:|
| Accuracy | 0.7284 | 0.7155 | −0.0129 |
| Precision | 0.6444 | 0.5891 | **−0.0553** |
| Recall | 0.6517 | **0.8539** | **+0.2022** |
| F1 | 0.6480 | **0.6972** | **+0.0492** |
| False positives | 32 | 53 | +21 |
| False negatives | 31 | 13 | −18 |

The original confusion matrix was `[[111,32],[31,58]]`; the oversampled confusion matrix was `[[90,53],[13,76]]`. **Recall increased the most**, by **20.22 percentage points**. For the retention team, this means **18 fewer genuinely lapsing customers missed**, at the cost of **21 additional non-lapsing customers targeted**. Precision declined, so increased coverage must be balanced against campaign costs. **B.Q7 validation passed.**

### B.Q8 — Interpret, save and use the selected model

**(a) Random Forest's five most important features:**

| Rank | Feature | Importance | One-line business interpretation |
|---:|---|---:|---|
| 1 | `monetary` | **0.280428** | Historical customer spending is a strong predictor of lapse in this fitted forest. |
| 2 | `frequency` | **0.277911** | Past ordering frequency helps distinguish likely lapsers from active customers. |
| 3 | `recency` | **0.109118** | Time since the last completed order contributes to predicted lapse risk. |
| 4 | `avg_delivery_minutes` | **0.059732** | Historical delivery experience provides additional predictive information. |
| 5 | `cancellation_rate` | **0.051834** | Historical cancellation behaviour helps the model distinguish customers. |

Monetary and frequency together account for **55.83%** of the forest's reported impurity-based feature importance. Importances do **not** indicate effect direction, causality, or the proportion of real-world lapse behaviour explained.

**(a) Logistic Regression coefficient signs:**

| Standardized numeric feature | Coefficient | Sign | One-line business reading |
|---|---:|---|---|
| `recency` | **+0.039167** | Positive | Greater time since last completed order is associated with slightly higher predicted lapse odds, other model inputs held fixed. |
| `frequency` | **−1.697333** | Negative | More historical completed orders are associated with lower predicted lapse odds. |
| `monetary` | **+0.283432** | Positive | Higher historical spending is conditionally associated with higher predicted lapse odds; this is not evidence that spending causes lapsing. |

**(b) Model persistence:** The selected **Random Forest preprocessing-and-model pipeline** was saved with `joblib.dump()` to `outputs/lapse_model.joblib` and reloaded using `joblib.load()`. The reloaded model reproduced the original **predictions and probabilities exactly for all 232 test customers**. **B.Q8(b) passed.**

**(c) Premium retention recommendations:** Of **40 Premium customers in the test set**, the **10 highest predicted lapse probabilities** were saved to `outputs/retention_list.csv` with the exact columns `customer_id`, `lapse_probability`, and `actually_lapsed`.

| Rank | Customer | Predicted lapse probability | Actually lapsed |
|---:|---|---:|---:|
| 1 | C0862 | 0.5105 | 0 |
| 2 | C0267 | 0.4772 | 0 |
| 3 | C1008 | 0.4733 | 0 |
| 4 | C0060 | 0.4717 | 0 |
| 5 | C0802 | 0.4716 | 0 |
| 6 | C0942 | 0.4652 | 0 |
| 7 | C0553 | 0.4628 | 0 |
| 8 | C0021 | 0.4613 | 0 |
| 9 | C0133 | 0.4607 | 0 |
| 10 | C0565 | 0.4583 | **1** |

**Actual outcome: 1 of 10 recommended customers truly lapsed (C0565).** The model's ranked probabilities for these ten customers ranged from **0.4583 to 0.5105**. The list was produced and validated as specified, but the observed 1/10 historical outcome should not be presented as prospective campaign effectiveness.

**Required limitation:** The Premium retention list is illustrative and not a leakage-free evaluation because Premium membership was derived from whole-window RFM data that includes the September lapse-label period; the lapse model's own features use only pre-cutoff information.

**B.Q8 validation passed:** feature interpretations reported, saved model reloaded with identical predictions, retention CSV verified, actual lapse count reported, and selected model F1 ≥ 0.55. **All Part B questions B.Q1–B.Q8 completed according to the supplied execution log.**

### Part B — Acceptance criteria and leakage boundaries

- **Reproducibility and deliverables:** The question-wise Part B execution logs reported passing validation for B.Q1–B.Q8. Required outputs are `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, and `outputs/retention_list.csv`. Re-run from a clean clone to confirm all artifacts are reproduced.
- **Historical prediction cutoff:** The lapse feature cutoff is **2025-09-01 00:00:00**. The September 1–30 period is used to create the future lapse label, **not model features**. No completed orders dated September 1 or later are used to construct the lapse model's input features. B.Q3 segment labels and Premium flags are **not** lapse-model features.
- **Training/test isolation:** `StandardScaler`, `OneHotEncoder` and any resampling are fitted/applied using **training data only**; neither the scaler nor encoder sees test rows during fitting. Five-fold CV refits preprocessing within each training fold. Class oversampling occurs only on training rows, never the test set or a validation fold. The 232-row test set remains untouched for final evaluation.
- **Performance and model choice:** The selected Random Forest attained **test F1 = 0.6369**, exceeding the required 0.55. It was selected on training CV F1 (0.6866), even though Logistic Regression later recorded a slightly higher test F1 (0.6480). The test result was not used to change the selection.
- **Interpretation and persistence:** The saved Random Forest pipeline reproduced predictions and probabilities after reload for all 232 test customers. Feature importance and coefficients describe associations, not causation.
- **Retention-list limitation:** The top-10 Premium retention list was produced and validated as requested; **1 of 10** actually lapsed. However, the Premium flag comes from whole-window segmentation, which includes the September label period. The retention shortlist is therefore **illustrative, not a leakage-free prospective targeting evaluation**. This caveat does **not** mean the lapse model's own historical feature matrix used future orders.

## Part C

Support Ticket Triage with Guardrails. C.Q1–C.Q7 were implemented and the submitted execution logs report passing checkpoints. The required exports are `outputs/triage_budget_demo.csv` and `outputs/triage_results.csv`.

### Part C — Question-wise code map

| Question | File | Main function(s) | Purpose |
|---|---|---|---|
| C.Q1 | `src/ticket_guardrails.py` | `run_ticket_guardrails()` | Reject empty tickets and flag overly long inputs before model calls. |
| C.Q2 | `src/ticket_redaction.py` | `run_ticket_redaction()` | Replace emails, Indian phone numbers and order IDs with safe placeholders. |
| C.Q3 | `src/ticket_prompts.py` | `run_ticket_prompts()`; `ZERO_SHOT_PROMPT`, `FEW_SHOT_PROMPT` | Define four allowed categories, JSON response instructions and three few-shot examples. |
| C.Q4 | `src/ticket_llm_retry.py` | `call_with_retry()`, `run_ticket_llm_retry()` | Assert prompt PII safety and retry transient failures with exponential backoff and jitter. |
| C.Q5 | `src/ticket_response_parser.py` | `parse_classification()`, `run_ticket_response_parser()` | Parse valid JSON, recover unambiguous keyword responses, and flag invalid outputs. |
| C.Q6 | `src/ticket_prompt_comparison.py` | `run_ticket_prompt_comparison()` | Compare zero-shot and few-shot classification accuracy and token costs. |
| C.Q7(a)–(b) | `src/ticket_triage.py` | `classify_with_budget()`, `route_tickets()`, `validate_saved_csv()`, `run_ticket_triage()` | Demonstrate budget exhaustion, perform final few-shot routing, export and reload both CSVs. |

**Independent execution:** Run `python -m src.ticket_triage` from the project root to reproduce both C.Q7 CSVs. This command runs C.Q1 and C.Q2 prerequisites, then C.Q7's own budget-aware calls. The master runner should call C.Q1 and C.Q2 once and pass the redacted tickets to C.Q6 and C.Q7; do not invoke the standalone C.Q4 and C.Q5 runners again because C.Q6 already uses their logic. The supplied `mock_llm.py` is not modified.

### C.Q1 — Input guardrails

Tickets are read in their original order. Empty ticket text is marked `skipped_empty`. A ticket exceeding **400 estimated input tokens** is marked `flagged_long`. Only `eligible` tickets can reach the LLM.

| Guardrail checkpoint | Observed result |
|---|---:|
| Total tickets | **164** |
| Eligible | **156** |
| Skipped empty | **4** |
| Flagged long | **4** |
| Estimated tokens for each long ticket | **734** |
| Maximum allowed estimated input tokens | **400** |

**Validation:** C.Q1 printed `C.Q1 ALL CHECKPOINTS PASSED`. All eight rejected tickets are excluded from model calls.

### C.Q2 — PII redaction

Eligible ticket text is sanitized before being included in model messages. The regex substitutions use `[EMAIL]`, `[PHONE]`, and `[ORDER_ID]` placeholders. The redaction stage preserves the C.Q1 statuses.

| Sensitive pattern | Substitutions |
|---|---:|
| Email addresses | **55** |
| Indian 10-digit phone numbers | **91** |
| Order IDs (`ORD-` followed by six digits) | **132** |
| **Total substitutions** | **278** |
| Remaining matches to defined PII patterns | **0** |

**Validation:** C.Q2 reported `C.Q2 ALL CHECKPOINTS PASSED`. Before an LLM call, C.Q4 additionally asserts that **both the system and user message** contain no matches to the defined PII patterns. These checks cover the specified regex patterns, not every conceivable form of personal data.

### C.Q3 — Zero-shot and few-shot prompts

Both prompts constrain the output to a JSON object with a `category` from **`billing`, `delivery`, `technical`, `fraud`**. The zero-shot prompt gives instructions without examples. The few-shot prompt includes **three labelled examples**, including fraud, to demonstrate the intended classification behaviour. Prompt text is defined centrally in `src/ticket_prompts.py`; static prompt wording is reviewed in code rather than checked by unnecessary runtime assertions.

### C.Q4 — Resilient LLM calls and retry logging

The model wrapper catches `TransientLLMError` and retries with exponential backoff, small random jitter, and at most **three retries**. Each retry prints the ticket ID, retry number and jittered wait; `retry=0` denotes the first retry after the initial failed attempt. The wrapper sends only the already-redacted user text, after asserting both model messages are free of the specified PII patterns.

| Retry checkpoint | Observed result |
|---|---:|
| Eligible calls completed in full run | **156** |
| Transient failures recovered | **19** |
| Permanent failures | **0** |
| Skipped/long tickets sent to model | **0** |

The budget demonstration stops early and therefore has **9** retry events; the full few-shot C.Q7 run logs **19** recovered retries. C.Q4's full-run validation reported `C.Q4 ALL CHECKPOINTS PASSED`.

### C.Q5 — Strict JSON parsing and keyword fallback

`parse_classification()` first parses JSON. A valid JSON object is accepted only when its `category` is one of the four allowed values. If JSON parsing fails, keyword fallback tokenizes the response and accepts it only when **exactly one** allowed category is present. A syntactically valid JSON response with an invalid category is **not** rescued by keyword fallback; ambiguous or unusable responses become `flagged_invalid_output`.

| Full-run outcome | Tickets |
|---|---:|
| `classified_json` | **135** |
| `classified_keyword` | **12** |
| `flagged_invalid_output` | **9** |
| `skipped_empty` | **4** |
| `flagged_long` | **4** |
| **Total** | **164** |

Thus **147** tickets receive an allowed category; **17** are reserved for human review in the unrestricted run. The first 147 are **rule-validated model classifications**, not independently human-confirmed labels.

### C.Q6 — Zero-shot versus few-shot evaluation

The two prompt configurations were evaluated against the supplied `true_category` labels on the **147 tickets** with an accepted classification. Accuracy is calculated only over classified tickets; skipped, long and invalid outputs are reported separately, not counted as correct predictions.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | **147** | **147** |
| Correct classifications | **126** | **147** |
| Accuracy among classified | **85.71%** | **100.00%** |
| Total actual model tokens | **12,364** | **21,724** |
| Tokens per classified ticket | **84.11** | **147.78** |
| Incorrect classifications | **21** | **0** |

In the zero-shot run, **11 fraud** tickets and **10 technical** tickets were incorrectly labelled billing; the few-shot run correctly classified these cases in the mock evaluation. Few-shot therefore improved classified-ticket accuracy by **14.29 percentage points**, using **9,360 additional tokens** (**75.7% more total tokens**), or approximately **63.67 additional tokens per classified ticket**. For this synthetic mock workload, the added token cost avoided 21 errors. The 100% result is specific to this supplied dataset and mock LLM; it is not a general production accuracy guarantee.

### C.Q7(a) — Zero-shot token-budget demonstration

The model budget is checked **before each call**: if the cumulative actual `response.usage.total_tokens` is already at least the budget, the eligible ticket is marked `budget_exhausted`, consumes **zero additional tokens**, and is routed to `human_review` with `NORMAL` priority. Ticket processing preserves source order. The last permitted response may bring the total slightly above the threshold; no future call is made once the threshold is reached.

| Budget-demo checkpoint | Observed result |
|---|---:|
| Prompt | **Zero-shot** |
| Configured token budget | **6,000** |
| Actual tokens consumed | **6,038** |
| Tickets marked `budget_exhausted` | **80** |
| Minimum required exhausted tickets | **1** |
| Retry events before budget stopping | **9** |
| Export | `outputs/triage_budget_demo.csv` |

The 6,000-token budget was sufficient to demonstrate exhaustion; no reduction to 5,000 was necessary. The output was **saved, reloaded and validated**. The validation asserts that exhausted tickets have `tokens_used=0`, route to human review and receive NORMAL priority.

### C.Q7(b) — Final few-shot classification and routing

The final run resets the mock LLM, uses the **few-shot prompt** and a **100,000-token budget**, and joins Part B's `outputs/customer_segments.csv` by `customer_id`. A customer missing from the segment file is treated as **not premium**. The priority of routing rules is deliberate:

| Priority order | Ticket condition | Route | Priority |
|---:|---|---|---|
| 1 | `skipped_empty`, `flagged_long`, `flagged_invalid_output`, or `budget_exhausted` | `human_review` | `NORMAL` |
| 2 | Classified with category `fraud`, including Premium customers | `security_team` | `HIGH` |
| 3 | Other classified category and `is_premium=true` | `premium_desk` | `HIGH` |
| 4 | Other classified category, non-premium or unknown customer | `support_queue` | `NORMAL` |

The final CSV uses **exactly these columns in order**: `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. Unavailable category and segment values are exported as blank cells.

| Final-run checkpoint | Observed result |
|---|---:|
| Total rows / unique tickets | **164 / 164** |
| Classified JSON / keyword | **135 / 12** |
| Invalid / empty / long | **9 / 4 / 4** |
| Budget-exhausted tickets | **0** |
| Actual tokens consumed | **21,724** |
| Token budget | **100,000** |
| Recovered transient failures | **19** |
| Export | `outputs/triage_results.csv` |

**Final routing distribution:**

| Route | Tickets |
|---|---:|
| `support_queue` | **102** |
| `security_team` | **34** |
| `human_review` | **17** |
| `premium_desk` | **11** |
| **Total** | **164** |

**Validation:** Both CSVs were written and reloaded. The program reported passing checks for required column order, 164 unique ticket IDs, reference status counts, token totals, budget-exhausted handling, fraud precedence, Premium routing, non-premium routing and human-review priority. Console output ended with **`C.Q7(a) AND C.Q7(b) ALL CHECKPOINTS PASSED`**.

**Part C acceptance summary:** No empty/long ticket reaches the LLM; specified PII patterns are checked before model calls; retry logs identify ticket, retry number and wait; 19 full-run transient failures are recovered; strict parsing yields 135 JSON, 12 keyword and 9 invalid outputs; zero-shot and few-shot accuracies are 85.71% and 100% on classified tickets; the zero-shot demo exhausts 80 tickets; and the final few-shot export contains 164 uniquely routed tickets. **Rule-validated classification means an output satisfied the parser and allowed-category rules, not that a human reviewed its correctness.** The 17 skipped/long/invalid tickets require human review, while classified tickets remain subject to model-error risk.


## Part D

### Policy RAG Assistant and Refund Agent

Part D combines a TF-IDF policy retrieval assistant with a tool-using ReAct refund agent. All model interactions use the supplied offline `mock_llm.py`; no API key or external model service is required. Refund amounts are in INR.

### Part D — Question-wise code map

| Question | Implementation | Evidence |
|---|---|---|
| D.Q1–D.Q2 | `src/policy_retrieval.py` | Policy chunking, overlap and TF-IDF retrieval |
| D.Q3–D.Q4 | `src/policy_assistant.py`, `src/policy_evaluation.py` | Grounded answers, fallback and evaluation |
| D.Q5 | `src/refund_tools.py` | Tool definitions, SQLite access and refund ledger |
| D.Q6 | `src/refund_agent.py` | Seven ReAct scenario runs and JSONL output |
| D.Q7 | `src/refund_safety_demo.py` | Iteration, token and timeout limits |
| D.Q8 | `src/refund_agent.py`, `src/refund_resume_demo.py` | Checkpoint and crash/resume verification |
| D.Q9 | This README | ReAct versus plan-and-execute comparison |

### D.Q1–D.Q4 — Policy retrieval, grounded answering and evaluation

The policy retriever splits policy content into **40-word chunks with 10-word overlap**, yielding **15 chunks**, and ranks relevant passages with TF-IDF. The assistant uses retrieved policy evidence and falls back when the answer is unsupported.

| Retrieval configuration | Retrieval checks | Factual-answer checks | Fallback checks |
|---|---:|---:|---:|
| 40-word chunks, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40-word chunks, 0-word overlap | 12/12 | 10/12 | 2/2 |

The 10-word overlap preserved context at chunk boundaries and produced better factual-answer accuracy in the measured evaluation. The zero-overlap run is a comparison, not the selected configuration.

### D.Q5 — Refund tools and safeguards

The agent exposes five tools:

| Tool | Purpose |
|---|---|
| `get_order` | Retrieve order information from the Part A SQLite database |
| `search_policy` | Retrieve the applicable refund policy |
| `calculate_refund` | Determine refund eligibility and amount |
| `issue_refund` | Issue an eligible refund and record it in the ledger |
| `escalate_to_human` | Hand off cases requiring manual review |

`src/refund_tools.py` defines `TOOLS_SPEC`, `TOOL_MAP` and `execute_tool`. Refund actions are recorded in `outputs/refund_ledger.jsonl`. The ledger supports idempotent repeat requests for an identical action and blocks conflicting duplicates. **Ten tool tests passed.**

### D.Q6 — ReAct agent: seven scenarios

The agent reasons over each tool observation before choosing the next action. The following are measured results from `python -m src.refund_agent`, not estimates.

| Scenario | Final status | Refund (INR) | Model calls | Tokens used | Check |
|---|---|---:|---:|---:|---|
| S1 | `REFUND` | 128.25 | 5 | 1,147 | PASS |
| S2 | `REFUND` | 135.00 | 5 | 1,147 | PASS |
| S3 | `ESCALATE` | 1,999.00 | 5 | 1,134 | PASS |
| S4 | `DENIED` | 0.00 | 4 | 805 | PASS |
| S5 | `DENIED` | 0.00 | 3 | 487 | PASS |
| S6 | `NOT_FOUND` | 0.00 | 2 | 172 | PASS |
| S7 | `FALLBACK_MAX_ITERATIONS` | 0.00 | 6 | 3,300 | PASS |

**Result:** `D.Q6 ALL SEVEN SCENARIOS PASSED`. The run writes seven records to `outputs/agent_runs.jsonl`. S3 is an escalation rather than an automatically issued refund; S7 demonstrates the maximum-iterations fallback.

### D.Q7 — Stop safely

The safety demonstration checks three independent limits and returns a human-handoff response instead of continuing indefinitely.

| Check | Final status | Model calls | Tokens used | Completed steps | Result |
|---|---|---:|---:|---:|---|
| D.Q7(a) Maximum iterations | `FALLBACK_MAX_ITERATIONS` | 6 | 3,300 | 6 | PASS |
| D.Q7(b) Token limit | `FALLBACK_TOKEN_LIMIT` | 2 | 202 | 1 | PASS |
| D.Q7(c) Timeout | `FALLBACK_TIMEOUT` | 0 | 0 | 0 | PASS |

The respective final messages are `Human handoff: maximum agent iterations reached.`, `Human handoff: agent token limit exceeded.`, and `Human handoff: agent time limit reached.`

**Result:** `D.Q7 ALL THREE SAFETY CHECKS PASSED`.

### D.Q8 — Checkpoint and resume after a simulated crash

After **every tool observation** is appended to the message history, the agent saves a checkpoint in `state/<run_id>.json`. Each checkpoint includes `run_id`, `scenario_id`, the full `messages` list, `tokens_used`, `step_number`, `tools_executed`, and additional execution state (`steps` and `llm_calls`). `resume_agent(run_id)` loads that checkpoint and continues from the saved state.

The S1 demonstration deliberately raises a simulated crash **immediately after the second tool execution and after its checkpoint is saved**.

| Execution order | Stage | Tool |
|---:|---|---|
| 1 | Before crash | `get_order` |
| 2 | Before crash | `search_policy` |
| — | Crash | Checkpoint saved after tool 2 |
| 3 | After `resume_agent(run_id)` | `calculate_refund` |
| 4 | After `resume_agent(run_id)` | `issue_refund` |

**Verified checkpoint:** `state/s1_crash_resume_demo.json`.

| Verified outcome | Result |
|---|---|
| Total tool executions | **4** |
| Repeated tools | **0** |
| Final status | **`REFUND`** |
| Refund amount | **INR 128.25** |
| Cumulative model calls | **5** |
| Cumulative tokens used | **1,147** |
| Demonstration result | **`D.Q8 CHECKPOINT AND RESUME PASSED`** |

The tool sequence across both parts of the run is exactly `get_order` → `search_policy` → `calculate_refund` → `issue_refund`. No completed tool was repeated after resumption.

### D.Q9 — ReAct versus plan-and-execute

Our ReAct refund agent follows a **reason–act–observe** cycle, using a model call after each tool observation to select the next action. Measured results were **S1: 5 calls / 1,147 tokens (`REFUND`)**, **S5: 3 calls / 487 tokens (`DENIED`)**, and **S6: 2 calls / 172 tokens (`NOT_FOUND`)**.

A **plan-and-execute** design would request the entire tool sequence in one initial model call, executing that plan and replanning only on error. It could reduce model calls and tokens for predictable cases, although actual savings depend on prompt size, plan complexity and replanning frequency. These savings are hypothetical; we did not benchmark a separate plan-and-execute implementation.

**Unexpected observations:** ReAct examines each response before acting again. When an order is `NOT_FOUND` (S6), it stops rather than attempting a refund; when an order is cancelled (S5), it denies the request without issuing a refund. A plan-and-execute system must explicitly validate each observation and interrupt or revise its plan to avoid executing inappropriate downstream steps.

**Choice:** We prefer **ReAct** for this refund workflow because eligibility depends on current order status, policy and refund limits. Its additional model calls provide useful step-by-step adaptability and auditability for financial actions. Deterministic tool-side validation and idempotency remain necessary regardless of agent architecture.

### Part D — Acceptance summary

The policy retrieval and fallback checks passed; all ten refund-tool tests passed; all seven ReAct scenarios passed; all three stop-safely demonstrations passed; and S1 recovered from a saved checkpoint with exactly four tool executions, no repeats and a final refund of INR 128.25. Required Part D artifacts are `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl` and the `state/` directory.


### Consolidated execution evidence from `python run_project.py`

The figures below are taken from the supplied `log.txt` generated by the complete project runner. They summarize representative results; the question-specific sections above contain the detailed interpretation.

| Part | Recorded execution outcome |
|---|---|
| A — Data Foundation | Audited 1,200 customers, 40 products and 13,084 raw orders. Saved 12,309 cleaned orders, including 11,110 completed orders. All seven SQL query checkpoints passed; pandas and SQL agreed on completed-order revenue of **INR 2,998,668.80**. |
| B — Customer Intelligence | Created RFM records for 1,175 customers, selected **K=4** (silhouette **0.526**), and built a lapse cohort of **1,156**, including **443 lapsed**. Selected Random Forest by training CV F1 (**0.6866**); test F1 was **0.6369**. Oversampled Logistic Regression test recall was **0.8539**, compared with **0.6517** before balancing. Saved the model and a ten-customer Premium retention list; **1/10** had actually lapsed. |
| C — Support Ticket Triage | Processed **164** tickets: **135** valid JSON classifications, **12** keyword recoveries, **9** invalid outputs, **4** empty and **4** long tickets. Zero-shot accuracy was **85.71%** (12,364 tokens); few-shot accuracy was **100%** (21,724 tokens), each on **147** classified tickets. The 6,000-token budget demo consumed **6,038** tokens and marked **80** tickets budget-exhausted. Final routing totals: **102** support, **34** security, **17** human review and **11** premium. |
| D — Policy RAG and Refund Agent | With 10-word overlap, policy retrieval and factual answering both scored **12/12** and fallback **2/2**; with zero overlap, factual answers scored **10/12**. All **seven** ReAct scenarios and **three** safety demonstrations passed. The D.Q8 simulated crash resumed after tool two, executed the remaining two tools once, and completed an **INR 128.25** refund. |

**Execution scope:** `python run_project.py` invokes the implemented Part A–D stages. D.Q1–D.Q3 are exercised through the policy evaluation; D.Q5 refund tools are exercised through the agent. The D.Q9 ReAct-versus-plan-and-execute comparison is written analysis, not an additional executable benchmark. The standalone `src.refund_tools` demonstration is not directly invoked by the master runner.

**Artifacts reported in the run:** `data/clean/customers.csv`, `data/clean/products.csv`, `data/clean/orders.csv`, `data/instamart.db`, `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`, `outputs/triage_budget_demo.csv`, `outputs/triage_results.csv`, `outputs/agent_runs.jsonl`, and `state/s1_crash_resume_demo.json`. The refund ledger `outputs/refund_ledger.jsonl` is an implementation artifact of `src.refund_tools`; its contents were not separately printed in this master-run log.

## Limitations

- The dataset and business scenario are synthetic; measured findings do not establish real-world Instamart customer behaviour.
- The Premium retention list is illustrative and is not a leakage-free evaluation because Premium membership was derived from whole-window RFM data, including the September lapse-label period. The lapse model's own features use only pre-cutoff information.
- The B.Q7 training-only oversampling comparison improves recall but increases false positives; outreach costs and missed-lapser costs have not been measured.
- Feature importance and Logistic Regression coefficients are associations, not causal explanations. Part A's observed completed-order correlations were very weak.
- Some exact changed-cell counts for cleaning rules, including city/gender normalization and cleared cancelled-order delivery values, were not printed in the recorded execution log. Those counts are not claimed.
- Part C's mock-model results and Part D's offline agent results may not generalize to live models or production support traffic.
- The plan-and-execute alternative in D.Q9 is a design comparison, not an independently implemented or benchmarked agent.

## Next Steps

The following are **future technical improvements** beyond the completed assignment. They are proposals, not completed work or reported outcomes.

### Part A — Data Foundation

- Extend the cleaning audit to record exact changed-cell and changed-row counts for each cleaning rule, including city, gender and cancelled-order delivery-time corrections.
- Investigate large-order outliers and additional data-quality anomalies while preserving legitimate high-value orders.

### Part B — Customer Intelligence

- Recompute Premium membership using only data available at the prediction cutoff, then evaluate retention targeting on a later unseen period.
- Compare retention thresholds and campaign costs against the costs of missed lapsers and unnecessary incentives.

### Part C — Support Ticket Triage

- Expand tests for ambiguous tickets, malformed model responses, privacy-sensitive inputs and retry failures.
- Evaluate the same guardrails and routing logic on a broader set of safely anonymized, real-world-like support examples.

### Part D — Policy RAG and Refund Agent

- Test retrieval quality and grounded responses with more policy documents and boundary cases.
- Implement plan-and-execute separately and benchmark it against ReAct using the same scenarios, safety limits, token budget and metrics.