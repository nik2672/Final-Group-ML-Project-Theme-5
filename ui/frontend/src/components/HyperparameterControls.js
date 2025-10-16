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
    { name: 'n_clusters', label: 'Number of Clusters', type: 'number', min: 2, max: 20, step: 1 },
    { name: 'max_k', label: 'Max K (Elbow Method)', type: 'number', min: 2, max: 20, step: 1 },
    { name: 'max_iter', label: 'Max Iterations', type: 'number', min: 100, max: 1000, step: 50 },
    { name: 'random_state', label: 'Random State', type: 'number', min: 0, max: 100, step: 1 },
  ],
  dbscan: [
    { name: 'eps', label: 'Epsilon (eps)', type: 'number', min: 0.1, max: 5.0, step: 0.1 },
    { name: 'min_samples', label: 'Min Samples', type: 'number', min: 2, max: 20, step: 1 },
  ],
  xgboost: [
    { name: 'n_estimators', label: 'Number of Estimators', type: 'number', min: 10, max: 500, step: 10 },
    { name: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.01, max: 0.5, step: 0.01 },
    { name: 'max_depth', label: 'Max Depth', type: 'number', min: 3, max: 15, step: 1 },
    { name: 'test_size', label: 'Test Size', type: 'number', min: 0.1, max: 0.4, step: 0.05 },
  ],
  arima: [
    { name: 'p', label: 'AR Order (p)', type: 'number', min: 0, max: 10, step: 1 },
    { name: 'd', label: 'Differencing (d)', type: 'number', min: 0, max: 3, step: 1 },
    { name: 'q', label: 'MA Order (q)', type: 'number', min: 0, max: 10, step: 1 },
    { name: 'sample_size', label: 'Sample Size', type: 'number', min: 1000, max: 100000, step: 1000 },
    { name: 'forecast_steps', label: 'Forecast Steps', type: 'number', min: 10, max: 200, step: 10 },
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
