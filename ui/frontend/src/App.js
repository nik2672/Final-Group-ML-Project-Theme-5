import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  AppBar,
  Toolbar,
  Alert,
  CircularProgress,
  Divider,
  LinearProgress,
} from '@mui/material';
import ModelSelector from './components/ModelSelector';
import HyperparameterControls from './components/HyperparameterControls';
import ResultsDisplay from './components/ResultsDisplay';
import DataStatusIndicator from './components/DataStatusIndicator';
import TargetMetricSelector from './components/TargetMetricSelector';
import RunHistory from './components/RunHistory';
import axios from 'axios';

function App() {
  const [selectedModel, setSelectedModel] = useState('');
  const [hyperparameters, setHyperparameters] = useState({});
  const [targetMetric, setTargetMetric] = useState('avg_latency');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [runHistory, setRunHistory] = useState([]);

  const handleModelChange = (model) => {
    setSelectedModel(model);
    setResults(null);
    setError(null);
    // Reset hyperparameters with defaults
    setHyperparameters(getDefaultHyperparameters(model));
  };

  const getDefaultHyperparameters = (model) => {
    const defaults = {
      'kmeans': { max_k: 8, max_iter: 300, random_state: 42 },
      'dbscan': { eps: 0.6, min_samples: 5 },
      'birch': { max_clusters: 7, threshold: 0.3, branching_factor: 50 },
      'optics': { min_samples: 5, max_eps: 2.0, xi: 0.1 },
      'hdbscan': { min_cluster_size: 8, min_samples: 5 },
      'xgboost': { n_estimators: 100, learning_rate: 0.1, max_depth: 6 },
      'arima': { p: 2, d: 1, q: 2, sample_size: 20000, forecast_steps: 50 },
      'sarima': { p: 1, d: 1, q: 1, seasonal_p: 1, seasonal_d: 1, seasonal_q: 1, seasonal_period: 24, sample_size: 20000, forecast_steps: 50 },
      'lstm': { units: 64, dropout: 0.2, lookback: 5, epochs: 10, batch_size: 64, max_samples: 100000 },
      'gru': { hidden_size: 48, lookback: 16, dropout: 0.1, epochs: 10, batch_size: 64, learning_rate: 0.001, max_features: 16, max_samples: 50000 },
    };
    return defaults[model] || {};
  };

  const handleHyperparameterChange = (name, value) => {
    setHyperparameters(prev => ({ ...prev, [name]: value }));
  };

  const handleRunModel = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await axios.post('/api/run-model', {
        model: selectedModel,
        hyperparameters: hyperparameters,
        target_metric: targetMetric,
      });
      setResults(response.data);

      // Add to history
      if (response.data.status === 'success') {
        setRunHistory(prev => [...prev, response.data]);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to run model. Check console for details.');
      console.error('Error running model:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClearHistory = () => {
    setRunHistory([]);
  };

  const handleCompareRuns = () => {
    alert('Comparison feature: Compare the metrics across your runs in the history table!');
  };

  const isForecastingModel = ['xgboost', 'arima', 'sarima', 'lstm', 'gru'].includes(selectedModel);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" component="div">
            5G Network Performance ML Runner
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, flex: 1 }}>
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 500 }}>
            Model Configuration
          </Typography>

          <Divider sx={{ my: 3 }} />

          <DataStatusIndicator />

          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
          />

          {selectedModel && (
            <>
              {isForecastingModel && (
                <Box sx={{ mt: 3 }}>
                  <TargetMetricSelector
                    value={targetMetric}
                    onChange={setTargetMetric}
                  />
                </Box>
              )}

              <Box sx={{ mt: 4 }}>
                <HyperparameterControls
                  model={selectedModel}
                  hyperparameters={hyperparameters}
                  onHyperparameterChange={handleHyperparameterChange}
                  onRunModel={handleRunModel}
                  loading={loading}
                />
              </Box>

              {error && (
                <Box sx={{ mt: 3 }}>
                  <Alert severity="error">{error}</Alert>
                </Box>
              )}

              {loading && (
                <Box sx={{ mt: 4 }}>
                  <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 2 }}>
                    Running {selectedModel.toUpperCase()} model... This may take 30s-2min
                  </Typography>
                  <LinearProgress />
                  <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                    <CircularProgress size={60} />
                  </Box>
                </Box>
              )}

              {results && !loading && (
                <Box sx={{ mt: 4 }}>
                  <ResultsDisplay results={results} model={selectedModel} />
                </Box>
              )}

              <RunHistory
                history={runHistory}
                onClear={handleClearHistory}
                onCompare={handleCompareRuns}
              />
            </>
          )}
        </Paper>
      </Container>

      <Box
        component="footer"
        sx={{
          py: 2,
          px: 2,
          mt: 'auto',
          backgroundColor: (theme) => theme.palette.grey[200],
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            5G Network Performance Analysis - ML Project Theme 5
          </Typography>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
