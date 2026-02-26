## Project Overview 
This project is a comprehensive Machine Learning web application built with Flask, designed to detect fraudulent credit card transactions. It bridges the gap between raw data science and practical banking operations. By utilizing advanced classification algorithms (XGBoost & Random Forest), the system provides a robust, real-time investigation tool tailored for bank fraud analysts.
##  Key Features
- **Three-Tier Risk Classification**
  - 🟢 Safe (Automatic Approval)
  - 🟡 Suspicious (Manual Review Required)
  - 🔴 Fraud ( Freeze & Bank Alert)

- **Dual-Model Architecture**
  - **XGBoost** (Primary production model – optimized with `scale_pos_weight`)
  - **Random Forest** (SMOTE-enhanced comparative model)
  - Dynamic model switching for performance comparison

- **Intelligent Alert System**
  - Email alerts triggered when:
    - `Fraud Probability ≥ 90%`
    - `Transaction Amount > $500`

- **Real-Time & Batch Processing**
  - Single transaction prediction
  - Bulk CSV file upload and analysis
- **Secure Configuration**
  - Environment variables (`.env`)
  - Email credentials excluded from version control
  ##  Decision Threshold Logic

| Model | Safe | Suspicious | Fraud |
|-------|------|------------|--------|
| **XGBoost** | Prob < 0.15 | 0.15 ≤ Prob < 0.83 | Prob ≥ 0.83 |
| **Random Forest** | Prob < 0.10 | 0.10 ≤ Prob < 0.36 | Prob ≥ 0.36 |
## Installation & Setup
  pip install -r requirements.txt
  create .env:
  EMAIL_USER=user_email@gmail.com 
  EMAIL_PASS=the_16_digit_app_password 
  BANK_EMAIL=bank_email@gmail.com
  Email credentials  are documented in the report
  Download dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
## Running the App 
    python app.py
    Then open http://127.0.0.1:5000 in your browser.