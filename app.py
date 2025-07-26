from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import json
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_model():
    # Create some sample data for training
    X = np.array([
        [100000, 75.5],  # [population, literacy]
        [200000, 80.0],
        [150000, 78.5],
        [300000, 85.0],
        [250000, 82.5]
    ])
    y = np.array([1, 1, 0, 1, 0])  # Binary classification (1 for urban, 0 for rural)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    with open('model/census_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Load or create model
try:
    model = pickle.load(open('model/census_model.pkl', 'rb'))
except FileNotFoundError:
    model = create_model()

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def process_census_data(df):
    # Basic statistics
    summary = {
        'total_districts': int(len(df)),
        'total_population': int(df['Population'].sum()),
        'avg_population': float(df['Population'].mean()),
        'avg_literacy': float(df['Literate'].mean() / df['Population'].mean() * 100),
        'max_population': int(df['Population'].max()),
        'min_population': int(df['Population'].min()),
        'total_households': int(df['Households'].sum()),
        'rural_households': int(df['Rural_Households'].sum()),
        'urban_households': int(df['Urban_Households'].sum())
    }
    
    # Prepare data for charts
    chart_data = {
        'labels': df['District name'].tolist(),
        'population': df['Population'].tolist(),
        'literacy_rate': (df['Literate'] / df['Population'] * 100).round(2).tolist(),
        'gender_distribution': {
            'male': int(df['Male'].sum()),
            'female': int(df['Female'].sum())
        },
        'education_levels': {
            'below_primary': int(df['Below_Primary_Education'].sum()),
            'primary': int(df['Primary_Education'].sum()),
            'middle': int(df['Middle_Education'].sum()),
            'secondary': int(df['Secondary_Education'].sum()),
            'higher': int(df['Higher_Education'].sum()),
            'graduate': int(df['Graduate_Education'].sum())
        },
        'religion_distribution': {
            'hindu': int(df['Hindus'].sum()),
            'muslim': int(df['Muslims'].sum()),
            'christian': int(df['Christians'].sum()),
            'sikh': int(df['Sikhs'].sum()),
            'buddhist': int(df['Buddhists'].sum()),
            'jain': int(df['Jains'].sum()),
            'others': int(df['Others_Religions'].sum())
        },
        'household_amenities': {
            'lpg_png': int(df['LPG_or_PNG_Households'].sum()),
            'electricity': int(df['Housholds_with_Electric_Lighting'].sum()),
            'internet': int(df['Households_with_Internet'].sum()),
            'computer': int(df['Households_with_Computer'].sum())
        }
    }
    
    # Convert all numpy types to Python native types
    summary = convert_numpy_types(summary)
    chart_data = convert_numpy_types(chart_data)
    
    return summary, chart_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process the data
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Process the data
            summary, chart_data = process_census_data(df)
            
            return jsonify({
                'summary': summary,
                'chart_data': chart_data
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        district = request.form['district']
        population = int(request.form['population'])
        literacy = float(request.form['literacy'])

        data = [[population, literacy]]
        result = model.predict(data)
        
        # Convert prediction to urban/rural
        prediction = "Urban" if result[0] > 0.5 else "Rural"

        return render_template('index.html', prediction=prediction, district=district)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
