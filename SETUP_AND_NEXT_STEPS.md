# NCAA Tournament App - Setup & Next Steps

## Project Structure Overview

```
ncaa-tournament-app/
├── backend/                    # Python Flask API
│   ├── models/                # Trained ML models (.pkl, .joblib)
│   ├── data/                  # Data files (.csv, .xlsx)
│   ├── src/
│   │   ├── app.py            # Flask API endpoints
│   │   ├── models.py         # Model loading & inference
│   │   ├── probabilities.py  # Win probability calculations
│   │   └── utils.py          # Helper functions
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile           # Container configuration
│   └── .env.template        # Environment variables template
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/      # Reusable components
│   │   ├── pages/          # Page components
│   │   ├── styles/         # CSS files
│   │   └── App.js          # Main app component
│   ├── package.json        # Node dependencies
│   └── Dockerfile         # Container configuration
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── GIT_SETUP_GUIDE.md    # This guide
└── docker-compose.yml    # Multi-container setup
```

## What We've Accomplished (Steps 1 & 2)

✅ Created complete project structure
✅ Set up backend skeleton with Flask API
✅ Created frontend React app with routing
✅ Designed all UI components (Banner, KPIs, Charts, Table)
✅ Added professional styling
✅ Created Docker configuration for deployment
✅ Set up Git ignore rules
✅ Created comprehensive documentation

## Next Steps - Phase by Phase

### PHASE 2: Model Development (Your Steps 1-4)

**Step 1: Build Composite Score Models**

Location: `backend/src/models.py`

You'll need to:
1. Load your trained models (XGBoost, scikit-learn)
2. Implement the `load_models()` function
3. Save your trained models to `backend/models/` directory

Example:
```python
def load_models(self):
    self.models['composite'] = joblib.load(self.models_dir / 'composite_model.pkl')
    self.models['tier'] = joblib.load(self.models_dir / 'tier_model.pkl')
    # etc...
```

**Step 2: Re-write Models into Functions**

Location: `backend/src/models.py`

Implement these functions:
- `predict_composite_score()`
- `predict_tier()`
- `predict_offense()`
- `predict_defense()`
- `predict_all()`
- `batch_predict()`

**Step 3: Apply Models to 2026 Teams**

Location: `backend/data/` and `backend/src/app.py`

1. Place your 2026 team data in `backend/data/2026_team_stats.csv`
2. Update the `/api/teams` endpoint in `app.py` to:
   - Load 2026 data
   - Run predictions
   - Return results

**Step 4: Calculate Probabilities**

Location: `backend/src/probabilities.py`

You have two options:

**Option A: Export/Import Workflow**
```python
# Export predictions
predictions_df = predictor.batch_predict(teams_2026)
predictions_df.to_csv('data/2026_predictions.csv')

# Calculate probabilities externally (your method)
# Then import back...
```

**Option B: Build Probability Functions**
Implement in `probabilities.py`:
- `calculate_head_to_head_probability()` - Win probability between two teams
- `calculate_heatmap_probabilities()` - Matrix for seeds 1-3
- `calculate_tournament_probabilities()` - Monte Carlo simulation
- `simulate_tournament()` - Full tournament simulation

### PHASE 3: Backend API Development

**Update API Endpoints** (backend/src/app.py)

For each endpoint, you'll need to:

1. **`/api/teams`** - Return all teams with predictions
```python
@app.route('/api/teams', methods=['GET'])
def get_teams():
    predictor = ModelPredictor()
    teams_data = load_data('data/2026_team_stats.csv')
    predictions = predictor.batch_predict(teams_data)
    return jsonify({'teams': predictions.to_dict('records')})
```

2. **`/api/kpis`** - Calculate and return 4 key metrics
3. **`/api/scatterplot`** - Return tier vs composite data
4. **`/api/probabilities/heatmap`** - Return win probability matrix

**Test the API:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/app.py

# Test in browser or with curl:
curl http://localhost:5000/api/health
```

### PHASE 4: Frontend Development

**No changes needed!** The frontend is already built and will automatically:
- Fetch data from your API
- Display visualizations
- Show the data table

Just start the frontend:
```bash
cd frontend
npm install
npm start
```

Visit: http://localhost:3000/women

### PHASE 5: Deployment to Google Cloud

We'll cover this once Phases 2-4 are complete.

## Immediate Action Items

### 1. Set Up Git (Use GIT_SETUP_GUIDE.md)
- Initialize repository
- Make first commit
- Connect to GitHub
- Push code

### 2. Set Up Python Environment
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add Your Models & Data
```bash
# Copy your trained models
cp /path/to/your/models/*.pkl backend/models/

# Copy your data
cp /path/to/your/data/*.csv backend/data/
```

### 4. Create Environment File
```bash
cd backend
cp .env.template .env
# Edit .env with your settings
```

### 5. Start Coding Your Models
Open `backend/src/models.py` in VS Code and start implementing!

## Development Workflow

1. **Daily Routine:**
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Activate virtual environment
   source backend/venv/bin/activate
   
   # Start backend
   cd backend
   python src/app.py
   
   # In another terminal, start frontend
   cd frontend
   npm start
   ```

2. **Testing:**
   - Backend: http://localhost:5000/api/health
   - Frontend: http://localhost:3000

3. **Making Changes:**
   - Edit files in VS Code
   - Test locally
   - Commit to Git
   - Push to GitHub

## File Organization Tips

### For Your Models:
```
backend/models/
├── composite_model.pkl      # Your composite score model
├── tier_model.pkl          # Your tier classification model
├── offense_model.pkl       # Offensive rating model
├── defense_model.pkl       # Defensive rating model
└── model_params.json       # Model hyperparameters (optional)
```

### For Your Data:
```
backend/data/
├── historical_tournament_data.csv  # Training data
├── historical_team_stats.csv       # Historical stats
├── 2026_team_stats.csv            # Current season data
└── 2026_seeds.csv                 # Tournament seeding
```

## Common Questions

**Q: Where do I put my existing model code?**
A: Adapt it into the functions in `backend/src/models.py`

**Q: My data files are too large for Git**
A: Use Git LFS or store in cloud (Google Drive, AWS S3)

**Q: How do I test the API without the frontend?**
A: Use curl, Postman, or visit the URL in your browser

**Q: Can I use Jupyter notebooks for development?**
A: Yes! Create a `notebooks/` folder and develop there, then move code to `src/`

**Q: What if I need additional Python packages?**
A: Add them to `requirements.txt` and run `pip install -r requirements.txt`

## Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **React Documentation**: https://react.dev/
- **Plotly Python**: https://plotly.com/python/
- **scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/

## Getting Help

When you need help with a specific step:
1. Reference this guide
2. Check the TODO comments in the code
3. Review the example structure provided
4. Ask for help with specific error messages

## Status Tracking

Use this to track your progress:

- [ ] Phase 1: Project Setup ✅ (COMPLETE)
- [ ] Phase 2: Model Development
  - [ ] Load and test models
  - [ ] Implement prediction functions
  - [ ] Apply to 2026 data
  - [ ] Calculate probabilities
- [ ] Phase 3: Backend API
  - [ ] Implement /api/teams
  - [ ] Implement /api/kpis
  - [ ] Implement /api/scatterplot
  - [ ] Implement /api/probabilities/heatmap
- [ ] Phase 4: Frontend Testing
  - [ ] Test locally
  - [ ] Verify visualizations
  - [ ] Check data table
- [ ] Phase 5: Deployment
  - [ ] Set up Google Cloud
  - [ ] Deploy backend
  - [ ] Deploy frontend
  - [ ] Configure domain

---

**You're ready to start Phase 2!** 

Begin by copying your models and data into the project, then start implementing the prediction functions in `backend/src/models.py`.

Good luck! 🏀
