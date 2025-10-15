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
} from '@mui/material';
import ModelSelector from './components/ModelSelector';
import HyperparameterControls from './components/HyperparameterControls';
import ResultsDisplay from './components/ResultsDisplay';
import axios from 'axios';

function App() {
  const [selectedModel, setSelectedModel] = useState('');
  const [hyperparameters, setHyperparameters] = useState({});
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleModelChange = (model) => {
    setSelectedModel(model);
    setResults(null);
    setError(null);
    // Reset hyperparameters with defaults
    setHyperparameters(getDefaultHyperparameters(model));
  };

  const getDefaultHyperparameters = (model) => {
    const defaults = {
      'kmeans': { n_clusters: 5, max_k: 10, max_iter: 300, random_state: 42 },
      'dbscan': { eps: 1.5, min_samples: 5 },
      'xgboost': { n_estimators: 100, learning_rate: 0.1, max_depth: 6, test_size: 0.2 },
      'arima': { p: 2, d: 1, q: 2, sample_size: 50000, forecast_steps: 50 },
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
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to run model. Check console for details.');
      console.error('Error running model:', err);
    } finally {
      setLoading(false);
    }
  };

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

          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
          />

          {selectedModel && (
            <>
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
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                  <CircularProgress size={60} />
                </Box>
              )}

              {results && !loading && (
                <Box sx={{ mt: 4 }}>
                  <ResultsDisplay results={results} model={selectedModel} />
                </Box>
              )}
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
