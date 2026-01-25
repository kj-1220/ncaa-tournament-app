# 🏀 NCAA Tournament App - Quick Start Guide

Welcome! This guide will get you started in **10 minutes**.

## What You Have

A complete full-stack web application with:
- ✅ Python Flask backend (ready for your ML models)
- ✅ React frontend (fully designed and styled)
- ✅ All UI components built
- ✅ API endpoints scaffolded
- ✅ Docker configuration
- ✅ Professional styling

## First Steps (Do These Now!)

### 1. Download the Project (2 minutes)

The project folder `ncaa-tournament-app` is ready to download.

### 2. Open in VS Code (1 minute)

1. Open VS Code
2. File → Open Folder
3. Select the `ncaa-tournament-app` folder

### 3. Set Up Git (5 minutes)

Open the terminal in VS Code (Terminal → New Terminal) and run:

```bash
# Navigate to project
cd ncaa-tournament-app

# Configure Git with YOUR information
git config --global user.name "kj-1220"
git config --global user.email "kjohnson1289@gmail.com"

# Initialize repository
git init

# Add all files
git add .

# First commit
git commit -m "Initial project setup"
```

**Connect to GitHub:**
- Click Source Control icon in VS Code (left sidebar)
- Click "Publish to GitHub"
- Sign in and publish

📖 **Detailed instructions:** See `GIT_SETUP_GUIDE.md`

### 4. Set Up Python Backend (2 minutes)

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## What to Do Next

### Immediate (Today):
1. ✅ Complete steps 1-4 above
2. 📁 Copy your trained models to `backend/models/`
3. 📁 Copy your 2026 data to `backend/data/`

### This Week:
4. 🔨 Implement model functions in `backend/src/models.py`
5. 🔨 Implement probability calculations in `backend/src/probabilities.py`
6. 🔨 Update API endpoints in `backend/src/app.py`

### Next Week:
7. 🧪 Test the backend API
8. 🎨 Test the frontend
9. 🚀 Deploy to Google Cloud

## File Locations

**Your models go here:**
```
backend/models/
├── composite_model.pkl
├── tier_model.pkl
├── offense_model.pkl
└── defense_model.pkl
```

**Your data goes here:**
```
backend/data/
├── 2026_team_stats.csv
├── 2026_seeds.csv
└── historical_data.csv
```

**Code you'll edit:**
```
backend/src/
├── models.py          ← Implement your ML models here
├── probabilities.py   ← Implement probability calculations
└── app.py            ← Update API endpoints
```

## Testing Your Work

### Test Backend:
```bash
cd backend
source venv/bin/activate
python src/app.py
```
Visit: http://localhost:5000/api/health

### Test Frontend:
```bash
cd frontend
npm install
npm start
```
Visit: http://localhost:3000

## Important Files to Read

1. **SETUP_AND_NEXT_STEPS.md** - Complete phase-by-phase guide
2. **GIT_SETUP_GUIDE.md** - Detailed Git & GitHub instructions
3. **README.md** - Project overview and documentation

## Code Examples to Get Started

### Example: Loading Your Model (models.py)

```python
def load_models(self):
    """Load all trained models from disk"""
    import joblib
    
    # Load your composite model
    self.models['composite'] = joblib.load(
        self.models_dir / 'composite_model.pkl'
    )
    
    # Load tier model
    self.models['tier'] = joblib.load(
        self.models_dir / 'tier_model.pkl'
    )
    
    print("Models loaded successfully!")
```

### Example: Making Predictions (models.py)

```python
def predict_composite_score(self, team_data):
    """Predict composite score for a team"""
    model = self.models['composite']
    
    # Assuming team_data is a DataFrame
    prediction = model.predict(team_data)
    
    return float(prediction[0])
```

### Example: API Endpoint (app.py)

```python
@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get all teams with predictions"""
    from src.models import ModelPredictor
    from src.utils import load_data
    
    # Load your 2026 data
    teams_df = load_data('data/2026_team_stats.csv')
    
    # Make predictions
    predictor = ModelPredictor()
    predictions = predictor.batch_predict(teams_df)
    
    return jsonify({
        'status': 'success',
        'teams': predictions.to_dict('records')
    })
```

## Getting Help

- **General questions:** See SETUP_AND_NEXT_STEPS.md
- **Git questions:** See GIT_SETUP_GUIDE.md
- **Code questions:** Check the TODO comments in each file

## Architecture Overview

```
Your User ←→ React Frontend ←→ Flask API ←→ Your ML Models
                (Port 3000)    (Port 5000)   (XGBoost/sklearn)
```

## What's Already Built For You

✅ Complete UI design
✅ Navigation sidebar
✅ Banner component
✅ 4 KPI cards
✅ Scatterplot visualization (Plotly)
✅ Heatmap visualization (Plotly)
✅ Sortable data table
✅ Responsive design
✅ Professional styling
✅ API structure
✅ Docker configuration

## What You Need to Build

🔨 Model loading functions
🔨 Prediction functions
🔨 Probability calculations
🔨 API endpoint implementations

---

## Ready to Start?

1. ✅ Download the project
2. ✅ Open in VS Code
3. ✅ Set up Git
4. ✅ Set up Python environment
5. 📖 Read SETUP_AND_NEXT_STEPS.md
6. 🔨 Start coding!

**You got this! 🚀**

The hardest part (project setup) is done. Now you can focus on what you do best - data science and modeling!
