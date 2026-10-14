from flask import Flask, render_template_string, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('naiveB.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template with modern, attractive embedded CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --border: #d1d5db;
        }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 35px 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 650px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            margin: 0 0 8px 0;
            color: var(--primary);
            font-size: 28px;
        }
        p.subtitle {
            margin: 0;
            color: var(--text-muted);
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 13px;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input, select {
            padding: 11px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            background-color: #f9fafb;
            transition: all 0.2s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        button:active {
            transform: scale(0.98);
        }
        .result-card {
            margin-top: 25px;
            padding: 18px;
            background-color: #e0e7ff;
            border-left: 5px solid var(--primary);
            border-radius: 8px;
            text-align: center;
            font-size: 17px;
            font-weight: 600;
            color: #312e81;
            animation: fadeIn 0.4s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            button {
                grid-column: span 1;
            }
            .container {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Employee Predictor</h2>
            <p class="subtitle">Enter employee parameters to run Naive Bayes classification</p>
        </div>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Education">Education</label>
                    <select id="Education" name="Education" required>
                        <option value="Bachelors">Bachelors</option>
                        <option value="Masters">Masters</option>
                        <option value="PHD">PHD</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="JoiningYear">Joining Year</label>
                    <input type="number" id="JoiningYear" name="JoiningYear" placeholder="e.g. 2018" required>
                </div>
                
                <div class="form-group">
                    <label for="City">City</label>
                    <select id="City" name="City" required>
                        <option value="Bangalore">Bangalore</option>
                        <option value="Pune">Pune</option>
                        <option value="New Delhi">New Delhi</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="PaymentTier">Payment Tier</label>
                    <select id="PaymentTier" name="PaymentTier" required>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" id="Age" name="Age" placeholder="e.g. 26" required>
                </div>
                
                <div class="form-group">
                    <label for="Gender">Gender</label>
                    <select id="Gender" name="Gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="EverBenched">Ever Benched</label>
                    <select id="EverBenched" name="EverBenched" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="ExperienceInCurrentDomain">Experience in Domain</label>
                    <input type="number" id="ExperienceInCurrentDomain" name="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
                </div>
                
                <button type="submit">Predict Status</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            Prediction Result: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract inputs from form
            education = request.form['Education']
            joining_year = float(request.form['JoiningYear'])
            city = request.form['City']
            payment_tier = float(request.form['PaymentTier'])
            age = float(request.form['Age'])
            gender = request.form['Gender']
            ever_benched = request.form['EverBenched']
            experience = float(request.form['ExperienceInCurrentDomain'])

            # Encode categorical features if your training set used label encoding. 
            # Adjust mapping dictionaries below to match your exact training label encoder mapping if needed.
            edu_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
            city_map = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
            gender_map = {'Male': 0, 'Female': 1}
            bench_map = {'No': 0, 'Yes': 1}

            # DataFrame matching the exact features list[cite: 1]
            input_data = pd.DataFrame([[
                edu_map.get(education, 0),
                joining_year,
                city_map.get(city, 0),
                payment_tier,
                age,
                gender_map.get(gender, 0),
                bench_map.get(ever_benched, 0),
                experience
            ]], columns=[
                'Education', 'JoiningYear', 'City', 'PaymentTier', 
                'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
            ])

            # Perform prediction
            pred = model.predict(input_data)[0]
            prediction = f"Class {pred}"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
