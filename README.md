#  Fraud Detection System

##  Project Overview
This project is a comprehensive Machine Learning web application built with Flask, designed to detect fraudulent credit card transactions. It bridges the gap between raw data science and practical banking operations. By utilizing advanced classification algorithms (XGBoost & Random Forest), the system provides a robust, real-time investigation tool tailored for bank fraud analysts.

##  Key Features
- **Two-Threshold Three-Tier Classification** :
 | Status | 🟢 SAFE | 🟡 SUSPICIOUS | 🔴 FRAUD |
| :--- | :--- | :--- | :--- |
| **Action** | Approve | Manual Review (Yellow Alert) | Auto-Freeze & Bank Alert |
| **XGBoost Logic** | Prob < 0.15 | 0.15 ≤ Prob < 0.83 | Prob ≥ 0.83 |
| **Random Forest Logic** | Prob < 0.10 | 0.10 ≤ Prob < 0.36 | Prob ≥ 0.36 |

- **Smart Email Alerts:** `Prob ≥ 90% AND Amount > $500`
- **Datasets:** `fraud_small_sample.csv` (website testing), `creditcard.csv` (model training)
- **Smart Email Alerts:** ONLY for `Probability >= 90% AND Amount > $500`
- **Single & Batch Processing:** Real-time analysis + bulk CSV upload
- **Secure:** `.env` file (Git ignored) for email credentials

##  Model Training & Insights
- **XGBoost** (primary, scale_pos_weight for imbalance)
- **Random Forest** (SMOTE preprocessing)
- **Threshold:** 0.83 for XBG/0.36 for RF 


##  Installation & Setup
```bash
git clone https://github.com/Batoul693/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
create .env:
EMAIL_USER=user_email@gmail.com
EMAIL_PASS=the_16_digit_app_password
BANK_EMAIL=bank_email@gmail.com
Download dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud