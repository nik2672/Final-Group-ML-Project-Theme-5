# 5G Network Performance ML UI

A simple, Material Design-based React UI for running machine learning models on 5G network performance data with configurable hyperparameters.

## Features

- **Model Selection**: Choose between K-Means, DBSCAN, XGBoost, and ARIMA models
- **Hyperparameter Tuning**: Adjust model hyperparameters via intuitive UI controls
- **Real-time Execution**: Run models directly from the browser
- **Results Display**: View metrics and output files after model execution
- **Material Design**: Clean, flat UI with straightforward minimalist UX

## Architecture

```
ui/
├── frontend/          # React + Material-UI
│   ├── src/
│   │   ├── App.js
│   │   └── components/
│   │       ├── ModelSelector.js
│   │       ├── HyperparameterControls.js
│   │       └── ResultsDisplay.js
│   └── package.json
└── backend/           # FastAPI server
    ├── main.py
    └── requirements.txt
```

## Prerequisites

- Node.js 16+ and npm
- Python 3.9+
- Processed data files in `data/` directory:
  - `features_for_clustering.csv`
  - `features_for_forecasting.csv`

**Important**: Run the feature engineering pipeline before using the UI:

```bash
python src/features/feature_engineering.py
```

## Installation

### 1. Install Backend Dependencies

```bash
cd ui/backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd ui/frontend
npm install
```

## Running the Application

You need to run both the backend and frontend servers.

### Terminal 1: Start Backend Server

```bash
cd ui/backend
python main.py
```

The API will be available at `http://localhost:8000`

### Terminal 2: Start Frontend Development Server

```bash
cd ui/frontend
npm start
```

The UI will open automatically at `http://localhost:3000`

## Using the UI

1. **Select a Model**: Choose from the dropdown (K-Means, DBSCAN, XGBoost, ARIMA)
2. **Configure Hyperparameters**: Adjust the model parameters using the input fields
3. **Run Model**: Click the "Run Model" button
4. **View Results**: Metrics and output files will be displayed below

## Available Models

### Clustering Models

**K-Means**
- `n_clusters`: Number of clusters (2-20)
- `max_k`: Maximum k for elbow method (2-20)
- `max_iter`: Maximum iterations (100-1000)
- `random_state`: Random seed (0-100)

**DBSCAN**
- `eps`: Epsilon neighborhood radius (0.1-5.0)
- `min_samples`: Minimum samples per cluster (2-20)

### Forecasting Models

**XGBoost**
- `n_estimators`: Number of trees (10-500)
- `learning_rate`: Learning rate (0.01-0.5)
- `max_depth`: Maximum tree depth (3-15)
- `test_size`: Test set size (0.1-0.4)

**ARIMA**
- `p`: AR order (0-10)
- `d`: Differencing order (0-3)
- `q`: MA order (0-10)
- `sample_size`: Data sample size (1000-100000)
- `forecast_steps`: Forecast horizon (10-200)

## API Endpoints

- `GET /`: Health check
- `POST /api/run-model`: Execute model with hyperparameters

Example request:

```json
{
  "model": "kmeans",
  "hyperparameters": {
    "n_clusters": 5,
    "max_iter": 300,
    "random_state": 42
  }
}
```

## Development Notes

- The UI uses a proxy configuration to forward API requests to the backend
- Both servers support hot-reload during development
- Results are computed in real-time (may take 30s-2min depending on model and data size)

## Troubleshooting

**"Clustering/Forecasting features not found"**
- Run feature engineering first: `python src/features/feature_engineering.py`

**Backend connection errors**
- Ensure backend is running on port 8000
- Check CORS configuration in `backend/main.py`

**Frontend build errors**
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`

## Technology Stack

- **Frontend**: React 18, Material-UI v5, Axios
- **Backend**: FastAPI, Uvicorn
- **ML Libraries**: scikit-learn, XGBoost, statsmodels
