import React from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Grid,
  Paper,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

const hyperparameterConfigs = {
  kmeans: [
    { name: 'max_k', label: 'Max K for Elbow Method', type: 'number', min: 5, max: 15, step: 1 },
    { name: 'max_iter', label: 'Max Iterations', type: 'number', min: 100, max: 1000, step: 50 },
    { name: 'random_state', label: 'Random State', type: 'number', min: 0, max: 100, step: 1 },
  ],
  dbscan: [
    { name: 'eps', label: 'Epsilon (eps)', type: 'number', min: 0.1, max: 1.0, step: 0.1 },
    { name: 'min_samples', label: 'Min Samples', type: 'number', min: 2, max: 10, step: 1 },
  ],
  birch: [
    { name: 'max_clusters', label: 'Max Clusters to Test', type: 'number', min: 5, max: 12, step: 1 },
    { name: 'threshold', label: 'Threshold', type: 'number', min: 0.1, max: 1.0, step: 0.1 },
    { name: 'branching_factor', label: 'Branching Factor', type: 'number', min: 20, max: 100, step: 10 },
  ],
  optics: [
    { name: 'min_samples', label: 'Min Samples', type: 'number', min: 2, max: 15, step: 1 },
    { name: 'max_eps', label: 'Max Epsilon', type: 'number', min: 0.5, max: 5.0, step: 0.1 },
    { name: 'xi', label: 'Xi (Cluster Selection)', type: 'number', min: 0.01, max: 0.2, step: 0.01 },
  ],
  hdbscan: [
    { name: 'min_cluster_size', label: 'Min Cluster Size', type: 'number', min: 3, max: 20, step: 1 },
    { name: 'min_samples', label: 'Min Samples (Optional)', type: 'number', min: 1, max: 15, step: 1 },
  ],
  xgboost: [
    { name: 'n_estimators', label: 'Number of Estimators', type: 'number', min: 10, max: 500, step: 10 },
    { name: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.01, max: 0.5, step: 0.01 },
    { name: 'max_depth', label: 'Max Depth', type: 'number', min: 3, max: 15, step: 1 },
  ],
  arima: [
    { name: 'p', label: 'AR Order (p)', type: 'number', min: 0, max: 10, step: 1 },
    { name: 'd', label: 'Differencing (d)', type: 'number', min: 0, max: 3, step: 1 },
    { name: 'q', label: 'MA Order (q)', type: 'number', min: 0, max: 10, step: 1 },
    { name: 'sample_size', label: 'Sample Size', type: 'number', min: 5000, max: 50000, step: 5000 },
    { name: 'forecast_steps', label: 'Forecast Steps', type: 'number', min: 10, max: 200, step: 10 },
  ],
  sarima: [
    { name: 'p', label: 'AR Order (p)', type: 'number', min: 0, max: 5, step: 1 },
    { name: 'd', label: 'Differencing (d)', type: 'number', min: 0, max: 2, step: 1 },
    { name: 'q', label: 'MA Order (q)', type: 'number', min: 0, max: 5, step: 1 },
    { name: 'seasonal_p', label: 'Seasonal AR (P)', type: 'number', min: 0, max: 3, step: 1 },
    { name: 'seasonal_d', label: 'Seasonal Diff (D)', type: 'number', min: 0, max: 2, step: 1 },
    { name: 'seasonal_q', label: 'Seasonal MA (Q)', type: 'number', min: 0, max: 3, step: 1 },
    { name: 'seasonal_period', label: 'Seasonal Period', type: 'number', min: 1, max: 168, step: 1 },
    { name: 'sample_size', label: 'Sample Size', type: 'number', min: 5000, max: 50000, step: 5000 },
    { name: 'forecast_steps', label: 'Forecast Steps', type: 'number', min: 10, max: 200, step: 10 },
  ],
  lstm: [
    { name: 'units', label: 'LSTM Units', type: 'number', min: 32, max: 128, step: 32 },
    { name: 'dropout', label: 'Dropout Rate', type: 'number', min: 0, max: 0.5, step: 0.1 },
    { name: 'lookback', label: 'Lookback Steps', type: 'number', min: 3, max: 20, step: 1 },
    { name: 'epochs', label: 'Epochs', type: 'number', min: 5, max: 20, step: 5 },
    { name: 'batch_size', label: 'Batch Size', type: 'number', min: 32, max: 128, step: 32 },
    { name: 'max_samples', label: 'Max Training Samples', type: 'number', min: 50000, max: 500000, step: 50000 },
  ],
  gru: [
    { name: 'hidden_size', label: 'Hidden Size', type: 'number', min: 16, max: 128, step: 16 },
    { name: 'lookback', label: 'Lookback Steps', type: 'number', min: 8, max: 32, step: 8 },
    { name: 'dropout', label: 'Dropout Rate', type: 'number', min: 0, max: 0.5, step: 0.1 },
    { name: 'epochs', label: 'Epochs', type: 'number', min: 5, max: 20, step: 5 },
    { name: 'batch_size', label: 'Batch Size', type: 'number', min: 16, max: 128, step: 16 },
    { name: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.0001, max: 0.01, step: 0.0001 },
    { name: 'max_features', label: 'Max Features', type: 'number', min: 8, max: 32, step: 4 },
    { name: 'max_samples', label: 'Max Training Samples', type: 'number', min: 10000, max: 100000, step: 10000 },
  ],
};

function HyperparameterControls({ model, hyperparameters, onHyperparameterChange, onRunModel, loading }) {
  const config = hyperparameterConfigs[model] || [];

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
        Hyperparameters
      </Typography>

      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          {config.map((param) => (
            <Grid item xs={12} sm={6} key={param.name}>
              <TextField
                fullWidth
                label={param.label}
                type="number"
                value={hyperparameters[param.name] || ''}
                onChange={(e) => onHyperparameterChange(param.name, parseFloat(e.target.value))}
                inputProps={{
                  min: param.min,
                  max: param.max,
                  step: param.step,
                }}
                variant="outlined"
              />
            </Grid>
          ))}
        </Grid>
      </Paper>

      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          size="large"
          startIcon={<PlayArrowIcon />}
          onClick={onRunModel}
          disabled={loading}
          sx={{ px: 6, py: 1.5 }}
        >
          {loading ? 'Running Model...' : 'Run Model'}
        </Button>
      </Box>
    </Box>
  );
}

export default HyperparameterControls;
