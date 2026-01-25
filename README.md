# NCAA Tournament Prediction App

A full-stack web application for predicting NCAA Women's Basketball Tournament outcomes using machine learning models.

## Project Structure

```
ncaa-tournament-app/
├── backend/                 # Python Flask API
│   ├── models/             # Trained ML models (XGBoost, scikit-learn)
│   ├── data/               # Historical and current season data
│   ├── src/                # Source code
│   │   ├── models.py       # Model loading and inference
│   │   ├── probabilities.py# Win probability calculations
│   │   ├── app.py          # Flask API endpoints
│   │   └── utils.py        # Helper functions
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container
├── frontend/               # React application
│   ├── public/            # Static assets
│   ├── src/               # React source code
│   │   ├── components/    # Reusable components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API integration
│   │   └── styles/        # CSS/styling
│   ├── package.json       # Node dependencies
│   └── Dockerfile         # Frontend container
└── docker-compose.yml     # Multi-container orchestration
```

## Features

### Women's Tournament Page
- **KPI Dashboard**: 4 key performance indicators
- **Tier Analysis**: Scatterplot showing team tiers vs composite scores
- **Matchup Heatmap**: Win probabilities for top-seeded teams (1-3 seeds)
- **Team Rankings Table**: Comprehensive stats for all teams including:
  - Rank, Team, Region, Seed
  - Bracket Value, Tier Scores
  - Overall/Offense/Defense ratings
  - Tournament advancement probabilities (Champion, Championship, Final Four, Elite Eight, Sweet 16, Round 2)

### Navigation
- Summary
- Matchups
- Bracket
- Women's Tournament (current page)
- Model Performance

## Technology Stack

### Backend
- **Language**: Python 3.10
- **Framework**: Flask
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy

### Frontend
- **Framework**: React
- **Visualizations**: Plotly.js / D3.js
- **Styling**: CSS/Tailwind CSS
- **HTTP Client**: Axios

### Deployment
- **Platform**: Google Cloud Platform (Cloud Run)
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 16+
- Docker (for deployment)
- Google Cloud SDK (for deployment)

### Local Development

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Running with Docker
```bash
docker-compose up --build
```

## Deployment to Google Cloud

(Instructions will be added in Phase 5)

## Model Information

The application uses trained machine learning models to predict:
- Individual team composite scores
- Team tier classifications
- Head-to-head win probabilities
- Tournament advancement probabilities

Models are trained on historical NCAA tournament data using XGBoost and scikit-learn.

## Contributing

This is a personal project. For questions or suggestions, please open an issue.

## License

MIT License

## Author

[Your Name] - Data Scientist

---

**Status**: In Development
**Last Updated**: January 2026
