import os
import smtplib
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from io import BytesIO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
BANK_EMAIL = os.getenv("BANK_EMAIL")
def send_fraud_alert(amount, probability, model_used, transaction_time):
    sender_email = BANK_EMAIL
    receiver_email = EMAIL_USER
    app_password = EMAIL_PASS

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "🚨 URGENT: High-Risk Transaction Flagged - Account Frozen"

    body = f"""
HIGH-RISK FRAUD DETECTED
Priority: CRITICAL
Model Used: {model_used}
Amount: ${amount}
Fraud Probability: {probability}%
Time: {transaction_time}

The system triggered this alert based on the high-value rule:
Amount >= $500 AND Probability >= 90%
Immediate manual investigation is recommended.
Please login to the Admin Dashboard to review the case.
____________
AI Fraud Detection System
Internal Risk Monitoring Unit
"""

    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(message)
        server.quit()
        print("Fraud alert email sent successfully.")
    except Exception as e:
        print("Error sending email:", e)



app = Flask("Fraud Detection System")
app.secret_key = 'admin_secret_key_123'

# ===================== LOAD DATA & MODELS =====================
DATA_DIR = 'data'
try:
    scaler = joblib.load('scaler.pkl')
    rf_model = joblib.load('random_forest_final.pkl')
    xgb_model = joblib.load('XGboost_model_without_smote.pkl')
    rf_threshold = joblib.load('rf_threshold.pkl')
    xgb_threshold = joblib.load('XGboost_best_threshold.pkl')
    risk_map_df = pd.read_csv('data/hourly_fraud_risk_map.csv')
    RISK_MAP = risk_map_df.set_index('Hour')['Class'].to_dict()

    csv_path = os.path.join(DATA_DIR, 'fraud_small_sample.csv')
    df_sample = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.read_csv('fraud_small_sample.csv')
    print("✅ Models and sample loaded successfully")
except Exception as e:
    print(f"⚠️ Loading error: {e}")
    df_sample = pd.DataFrame()

COLS_XGB_RF = [f'V{i}' for i in range(1, 29)] + ['hour_sin', 'hour_cos', 'hourly_fraud_risk', 'Amount_scaled']

def prepare_data(raw_row_dict, risk_map=RISK_MAP):
    df = pd.DataFrame([raw_row_dict])

    # Scale Amount if missing
    if 'Amount_scaled' not in df.columns:
        df['Amount_scaled'] = scaler.transform(df[['Amount']]) if 'Amount' in df.columns else 0.0

    # Create Hour column if missing
    if 'Hour' not in df.columns and 'Time' in df.columns:
        df['Hour'] = (df['Time'] // 3600) % 24

    # Create cyclical hour features
    if 'Hour' in df.columns:
        h = df['Hour'].iloc[0]
        df['hour_sin'] = np.sin(2 * np.pi * h / 24)
        df['hour_cos'] = np.cos(2 * np.pi * h / 24)

        #  Add hourly fraud risk feature from preloaded map
        df['hourly_fraud_risk'] = risk_map.get(h, 0.0)
    else:
        # Hour column missing → default risk = 0.17
        df['hourly_fraud_risk'] = 0.17

    # Fill missing model columns
    for col in COLS_XGB_RF:
        if col not in df.columns:
            df[col] = 0.0

    return df[COLS_XGB_RF], int(df.get('Hour', [0]).iloc[0])


# ===================== LOGIN / LOGOUT =====================
@app.route('/', methods=['GET'])
def home():
    if session.get('logged_in'):
        return redirect(url_for('predict'))
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == '123':
            session['logged_in'] = True
            return redirect(url_for('predict'))
        else:
            error = 'Invalid credentials'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# ===================== SINGLE PREDICTION =====================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    results, error, selected_row = None, None, 0
    rows_indices = df_sample.index.tolist() if not df_sample.empty else []

    if request.method == 'POST':
        try:
            row_id = int(request.form.get('row_id', 0))
            selected_model = request.form.get('selected_model', 'RandomForest')
            selected_row = row_id

            final_input_df, hour = prepare_data(df_sample.iloc[row_id].to_dict())

            if selected_model == 'XGBoost':
                prob = float(xgb_model.predict_proba(final_input_df.values)[0][1])
                thresh = xgb_threshold
                medium_band = 0.15
            else:
                prob = float(rf_model.predict_proba(final_input_df)[0][1])
                thresh = rf_threshold
                medium_band = 0.10

            if prob >= thresh:
                status = "FRAUD"
            elif prob >= medium_band:
                status = "SUSPICIOUS"
            else:
                status = "SAFE"

            amount = df_sample.iloc[row_id].get('Amount', 0)
            prob_percent = round(prob * 100, 2)

            if (
                amount >= 500 and
                prob >= 0.90
            ):
                send_fraud_alert(
                    amount=amount,
                    probability=prob_percent,
                    model_used=selected_model,
                    transaction_time=hour
                )                

            results = [{
                'model': selected_model,
                'probability': prob,
                'prediction': status,
                'hour': hour,
                'amount': amount
            }]

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('predict.html', rows=rows_indices, results=results, error=error, selected_row=selected_row)

# ===================== BATCH PREDICTION =====================
@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    batch_results = []
    selected_model = request.form.get('selected_model', 'RandomForest')

    if not df_sample.empty:
        if selected_model == 'XGBoost':
            current_thresh = xgb_threshold
            medium_band = 0.15
        else:
            current_thresh = rf_threshold
            medium_band = 0.10

        for index, row in df_sample.head(1000).iterrows():
            try:
                input_df, hour = prepare_data(row.to_dict())
                if selected_model == 'XGBoost':
                    prob = float(xgb_model.predict_proba(input_df.values)[0][1])
                else:
                    prob = float(rf_model.predict_proba(input_df)[0][1])

                if prob >= current_thresh:
                    status = "FRAUD"
                elif prob >= medium_band:
                    status = "SUSPICIOUS"
                else:
                    status = "SAFE"

                amount = row.get('Amount', 0)
                prob_percent = round(prob * 100, 2)

                if (
                    amount >= 500 and
                    prob >= 0.90
                ):
                    send_fraud_alert(
                        amount=amount,
                        probability=prob_percent,
                        model_used=selected_model,
                        transaction_time=hour
                    )

                batch_results.append({
                    'id': index,
                    'amount': row.get('Amount', 0),
                    'hour': hour,
                    'probability': prob,
                    'prediction': status
                })
            except:
                continue

    hourly_data = [RISK_MAP.get(h, 0.17) for h in range(24)]
    return render_template('batch.html', results=batch_results, hourly_data=hourly_data, selected_model=selected_model)

# ===================== DOWNLOAD CSV =====================
@app.route('/download_csv')
def download_csv():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Use the latest batch results
    selected_model = request.args.get('model', 'RandomForest')
    batch_results = []
    if not df_sample.empty:
        if selected_model == 'XGBoost':
            current_thresh = xgb_threshold
            medium_band = 0.15
        else:
            current_thresh = rf_threshold
            medium_band = 0.10

        for index, row in df_sample.head(900).iterrows():
            try:
                input_df, hour = prepare_data(row.to_dict())
                if selected_model == 'XGBoost':
                    prob = float(xgb_model.predict_proba(input_df.values)[0][1])
                else:
                    prob = float(rf_model.predict_proba(input_df)[0][1])

                if prob >= current_thresh:
                    status = "FRAUD"
                elif prob >= medium_band:
                    status = "SUSPICIOUS"
                else:
                    status = "SAFE"

                batch_results.append({
                    'ID': index,
                    'Amount': row.get('Amount', 0),
                    'Hour': hour,
                    'Probability': prob,
                    'Prediction': status
                })
            except:
                continue

    df_csv = pd.DataFrame(batch_results)
    buffer = BytesIO()
    df_csv.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
    buffer,
    mimetype='text/csv',
    download_name='batch_results.csv',  
    as_attachment=True)



# ===================== DASHBOARD =====================
import json

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Load metrics from JSON
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)

    media_dir = os.path.join(app.static_folder, 'media')
    images = os.listdir(media_dir) if os.path.exists(media_dir) else []
    hourly_data = [RISK_MAP.get(h, 0.17) for h in range(24)]
    max_risk = max(hourly_data) if hourly_data else 0
    if max_risk == 0:
        max_risk = 1

    return render_template(
        'dashboard.html',
        images=images,
        metrics=metrics,
        hourly_data=hourly_data,
        max_risk=max_risk
    )


# ===================== ABOUT =====================
@app.route('/about')
def about():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('about.html')

# ===================== RUN APP =====================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
