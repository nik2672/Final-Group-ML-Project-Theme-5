# 5G Network Performance ML UI

A modern, comprehensive web interface for running and comparing machine learning models on 5G network performance data. Built with Material-UI and FastAPI, this application provides an intuitive way to experiment with clustering and forecasting models using configurable hyperparameters.

---

##  Features

### Core Functionality
- **10 ML Models**: Support for K-Means, DBSCAN, BIRCH, OPTICS, HDBSCAN (clustering) and XGBoost, ARIMA, SARIMA, LSTM, GRU (forecasting)
- **Interactive Hyperparameter Tuning**: Real-time configuration of model parameters through intuitive controls
- **Data Status Monitoring**: Live indicator showing feature availability and data readiness
- **Target Metric Selection**: Choose which metric to forecast (avg_latency, avg_throughput, etc.)
- **Run History**: Track and compare multiple model executions side-by-side
- **Results Visualization**: View metrics, performance charts, and output files directly in the browser

### User Experience
- **Material Design UI**: Clean, modern interface with responsive layout
- **Real-time Feedback**: Progress indicators and status updates during model execution
- **Error Handling**: Clear error messages and troubleshooting guidance
- **Persistent History**: Keep track of all runs during your session
- **Output Viewer**: Browse and download generated plots and result files

---

##  Project Structure

```
ui/
├── frontend/                      # React application
│   ├── src/
│   │   ├── App.js                # Main application component
│   │   └── components/
│   │       ├── ModelSelector.js           # Model dropdown
│   │       ├── HyperparameterControls.js  # Dynamic parameter inputs
│   │       ├── TargetMetricSelector.js    # Metric selection
│   │       ├── ResultsDisplay.js          # Metrics & results viewer
│   │       ├── OutputViewer.js            # File browser
│   │       ├── RunHistory.js              # Execution history table
│   │       └── DataStatusIndicator.js     # Feature availability status
│   ├── public/
│   └── package.json
├── backend/                       # FastAPI server
│   ├── main.py                   # API endpoints and model execution
│   └── requirements.txt
├── start_backend.sh              # Backend startup script
├── start_frontend.sh             # Frontend startup script
├── QUICKSTART.md                 # Quick setup guide
└── README.md                     # This file
```

---

##  Prerequisites

### Required Software
- **Node.js 16+** and npm
- **Python 3.9+**

### Required Data Files
Before using the UI, you **must** run the feature engineering pipeline to generate:
- `data/features_for_clustering.csv`
- `data/features_for_forecasting.csv`

Run the following command from the project root:

```bash
python src/features/feature_engineering.py
```

The UI includes a **Data Status Indicator** that shows whether these files are available.

---

##  Installation

### One-Time Setup

#### 1. Install Backend Dependencies
```bash
cd ui/backend
pip install -r requirements.txt
```

**Backend Dependencies:**
- FastAPI 0.104.1
- Uvicorn 0.24.0
- scikit-learn, XGBoost, statsmodels
- pandas, numpy, matplotlib

#### 2. Install Frontend Dependencies
```bash
cd ui/frontend
npm install
```

**Frontend Dependencies:**
- React 18.2
- Material-UI v5
- Axios
- React Scripts

*Note: `npm install` may take 2-5 minutes depending on your internet connection.*

---

##  Running the Application

### Quick Start (Recommended)

Use the provided startup scripts to launch both servers:

**Terminal 1 - Backend:**
```bash
cd ui
./start_backend.sh    # Linux/Mac
# or
.\start_backend.sh    # Windows (if using Git Bash)
```

**Terminal 2 - Frontend:**
```bash
cd ui
./start_frontend.sh   # Linux/Mac
# or
.\start_frontend.sh   # Windows (if using Git Bash)
```

### Manual Startup

**Terminal 1 - Backend Server:**
```bash
cd ui/backend
python main.py
```
Backend API will be available at `http://localhost:8000`

**Terminal 2 - Frontend Development Server:**
```bash
cd ui/frontend
npm start
```
UI will automatically open at `http://localhost:3000`

---

##  Using the UI

### Workflow

1. **Check Data Status** 
   - Look for the green "Data Ready" indicator in the top-right corner
   - If red, run the feature engineering pipeline first

2. **Select a Model**
   - Choose from the dropdown: K-Means, DBSCAN, BIRCH, OPTICS, HDBSCAN, XGBoost, ARIMA, SARIMA, LSTM, or GRU

3. **Configure Hyperparameters**
   - Adjust model parameters using sliders and input fields
   - Default values are pre-populated for each model

4. **Select Target Metric** (Forecasting Models Only)
   - Choose which metric to predict: avg_latency, avg_throughput, packet_loss_rate, etc.

5. **Run Model**
   - Click "Run Model" button
   - Monitor progress with the loading indicator
   - Execution time varies: 10s-5min depending on model and data size

6. **View Results**
   - **Metrics**: Performance scores, accuracy, error rates
   - **Output Files**: Generated plots and visualizations
   - **Run Details**: Execution time, hyperparameters used

7. **Compare Runs**
   - View run history table at the bottom
   - Compare metrics across different hyperparameter configurations
   - Clear history when starting a new experiment

---

##  Available Models

### Clustering Models

#### **K-Means**
Partitions data into k clusters using centroid-based grouping.

**Hyperparameters:**
- `max_k`: Maximum k for elbow method (2-20, default: 8)
- `max_iter`: Maximum iterations (100-1000, default: 300)
- `random_state`: Random seed for reproducibility (0-100, default: 42)

#### **DBSCAN**
Density-based clustering that identifies arbitrary-shaped clusters.

**Hyperparameters:**
- `eps`: Neighborhood radius (0.1-5.0, default: 0.6)
- `min_samples`: Minimum points per cluster (2-20, default: 5)

#### **BIRCH**
Balanced Iterative Reducing and Clustering using Hierarchies.

**Hyperparameters:**
- `max_clusters`: Target number of clusters (2-20, default: 7)
- `threshold`: Subcluster radius threshold (0.1-2.0, default: 0.3)
- `branching_factor`: CF tree branching factor (10-100, default: 50)

#### **OPTICS**
Ordering Points To Identify the Clustering Structure.

**Hyperparameters:**
- `min_samples`: Core point neighborhood size (2-20, default: 5)
- `max_eps`: Maximum distance between samples (0.5-5.0, default: 2.0)
- `xi`: Steepness parameter (0.01-0.5, default: 0.1)

#### **HDBSCAN**
Hierarchical DBSCAN for varying density clusters.

**Hyperparameters:**
- `min_cluster_size`: Minimum cluster size (2-50, default: 8)
- `min_samples`: Conservative core point threshold (1-20, default: 5)

---

### Forecasting Models

#### **XGBoost**
Gradient boosting regression for time series prediction.

**Hyperparameters:**
- `n_estimators`: Number of boosting rounds (10-500, default: 100)
- `learning_rate`: Step size shrinkage (0.01-0.5, default: 0.1)
- `max_depth`: Maximum tree depth (3-15, default: 6)

**Target Metrics:** avg_latency, avg_throughput, packet_loss_rate, jitter, signal_strength

#### **ARIMA**
AutoRegressive Integrated Moving Average model.

**Hyperparameters:**
- `p`: AR order (0-10, default: 2)
- `d`: Differencing order (0-3, default: 1)
- `q`: MA order (0-10, default: 2)
- `sample_size`: Data points to use (1000-100000, default: 20000)
- `forecast_steps`: Future steps to predict (10-200, default: 50)

#### **SARIMA**
Seasonal ARIMA with seasonal components.

**Hyperparameters:**
- `p`, `d`, `q`: Non-seasonal parameters (default: 1,1,1)
- `seasonal_p`, `seasonal_d`, `seasonal_q`: Seasonal parameters (default: 1,1,1)
- `seasonal_period`: Seasonality period (1-168, default: 24)
- `sample_size`: Data points (1000-100000, default: 20000)
- `forecast_steps`: Future predictions (10-200, default: 50)

#### **LSTM**
Long Short-Term Memory neural network.

**Hyperparameters:**
- `units`: LSTM layer neurons (16-256, default: 64)
- `dropout`: Dropout rate (0.0-0.5, default: 0.2)
- `lookback`: Time steps to look back (1-50, default: 5)
- `epochs`: Training iterations (5-100, default: 10)
- `batch_size`: Samples per gradient update (16-256, default: 64)
- `max_samples`: Maximum data points (1000-500000, default: 100000)

#### **GRU**
Gated Recurrent Unit neural network.

**Hyperparameters:**
- `hidden_size`: GRU layer size (16-256, default: 48)
- `lookback`: Historical window (1-50, default: 16)
- `dropout`: Regularization rate (0.0-0.5, default: 0.1)
- `epochs`: Training epochs (5-100, default: 10)
- `batch_size`: Batch size (16-256, default: 64)
- `learning_rate`: Optimizer learning rate (0.0001-0.1, default: 0.001)
- `max_features`: Feature subset size (1-50, default: 16)
- `max_samples`: Data limit (1000-500000, default: 50000)

---

##  API Endpoints

### **GET** `/`
Health check endpoint.

**Response:**
```json
{
  "message": "5G ML Model Runner API",
  "status": "running"
}
```

### **GET** `/api/data-status`
Check feature file availability.

**Response:**
```json
{
  "clustering_available": true,
  "forecasting_available": true
}
```

### **POST** `/api/run-model`
Execute a machine learning model.

**Request Body:**
```json
{
  "model": "kmeans",
  "hyperparameters": {
    "max_k": 8,
    "max_iter": 300,
    "random_state": 42
  },
  "target_metric": "avg_latency"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "model": "kmeans",
  "metrics": {
    "silhouette_score": 0.65,
    "davies_bouldin_score": 0.82,
    "optimal_k": 5
  },
  "execution_time": "12.34s",
  "output_files": ["elbow_plot.png", "cluster_distribution.png"],
  "timestamp": "2025-10-27T10:30:45"
}
```

---

##  Development

### Frontend Development
- **Hot Reload**: Changes auto-refresh at `http://localhost:3000`
- **Build Production**: `npm run build`
- **Proxy Config**: API requests forwarded to backend via proxy in `package.json`

### Backend Development
- **Auto-reload**: Uvicorn watches for file changes
- **API Docs**: Visit `http://localhost:8000/docs` for interactive API documentation
- **Logs**: Console shows request/response details and model execution progress

---

##  Troubleshooting

### "Data not ready" indicator shows red

**Solution:** Run feature engineering to generate required CSV files
```bash
python src/features/feature_engineering.py
```

### Backend connection errors / CORS issues

**Check:**
1. Backend is running on port 8000
2. Frontend proxy configuration in `package.json`
3. CORS settings in `backend/main.py` allow `http://localhost:3000`

### Model execution takes very long

**Tips:**
- Reduce `sample_size` or `max_samples` for forecasting models
- Use smaller `max_k` for clustering models
- Check terminal for progress logs

### Frontend build errors

**Solution:** Clear cache and reinstall dependencies
```bash
cd ui/frontend
rm -rf node_modules package-lock.json
npm install
```

### "Module not found" Python errors

**Solution:** Ensure backend is run from correct directory
```bash
cd ui/backend
python main.py
```

---

##  Technology Stack

### Frontend
- **React 18.2** - UI library
- **Material-UI v5** - Component library
- **Axios** - HTTP client
- **React Scripts** - Build tooling

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Machine Learning
- **scikit-learn** - K-Means, DBSCAN, BIRCH, OPTICS
- **hdbscan** - HDBSCAN clustering
- **XGBoost** - Gradient boosting
- **statsmodels** - ARIMA, SARIMA
- **TensorFlow/Keras** - LSTM, GRU
- **pandas, numpy** - Data processing
- **matplotlib** - Visualization

---

##  License

This project is part of the Final Group ML Project (Theme 5) for 5G network performance analysis.

---

##  Contributing

For questions or issues, please refer to the main project repository documentation or contact the development team.

---

##  Additional Resources

- **QUICKSTART.md** - Step-by-step setup guide
- **Backend API Docs** - `http://localhost:8000/docs` (when running)
- **Main Project README** - See parent directory for full project context
