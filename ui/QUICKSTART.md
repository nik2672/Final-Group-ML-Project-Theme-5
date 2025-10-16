# Quick Start Guide

## Prerequisites

Before running the UI, ensure you have:

1. **Feature-engineered data files** - Run the pipeline first:
   ```bash
   python src/features/feature_engineering.py
   ```

2. **Node.js 16+** and **Python 3.9+** installed

## Installation (One-time setup)

### Backend
```bash
cd ui/backend
pip install -r requirements.txt
```

### Frontend
```bash
cd ui/frontend
npm install
```

Note: `npm install` may take 2-5 minutes depending on your connection.

## Running the Application

### Option 1: Using startup scripts (Recommended)

**Terminal 1 - Backend:**
```bash
cd ui
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
cd ui
./start_frontend.sh
```

### Option 2: Manual startup

**Terminal 1 - Backend:**
```bash
cd ui/backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd ui/frontend
npm start
```

The UI will automatically open at `http://localhost:3000`

## First-time Usage

1. Select a model from the dropdown (e.g., K-Means)
2. Configure hyperparameters using the input fields
3. Click "Run Model"
4. Wait 30s-2min for results (varies by model and data size)
5. View metrics and output files

## Troubleshooting

**"Module 'fastapi' not found"**
- Backend dependencies not installed: `cd ui/backend && pip install -r requirements.txt`

**"Cannot find module 'react'"**
- Frontend dependencies not installed: `cd ui/frontend && npm install`

**"Clustering features not found"**
- Data not processed: `python src/features/feature_engineering.py`

**Port already in use**
- Kill existing processes on port 3000 (frontend) or 8000 (backend)
- macOS/Linux: `lsof -ti:3000 | xargs kill -9`
