import React from 'react';
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Chip,
} from '@mui/material';

const models = [
  {
    value: 'kmeans',
    label: 'K-Means Clustering',
    category: 'Clustering',
    description: 'Partitions network zones into k clusters based on performance metrics',
  },
  {
    value: 'dbscan',
    label: 'DBSCAN Clustering',
    category: 'Clustering',
    description: 'Density-based clustering that identifies outlier zones automatically',
  },
  {
    value: 'xgboost',
    label: 'XGBoost Forecasting',
    category: 'Forecasting',
    description: 'Gradient boosting for predicting network performance metrics',
  },
  {
    value: 'arima',
    label: 'ARIMA Forecasting',
    category: 'Forecasting',
    description: 'Time series forecasting using statistical methods',
  },
];

function ModelSelector({ selectedModel, onModelChange }) {
  const selectedModelData = models.find(m => m.value === selectedModel);

  return (
    <Box>
      <FormControl fullWidth variant="outlined">
        <InputLabel id="model-select-label">Select Model</InputLabel>
        <Select
          labelId="model-select-label"
          id="model-select"
          value={selectedModel}
          label="Select Model"
          onChange={(e) => onModelChange(e.target.value)}
        >
          <MenuItem value="">
            <em>Choose a model...</em>
          </MenuItem>
          {models.map((model) => (
            <MenuItem key={model.value} value={model.value}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip label={model.category} size="small" color="primary" variant="outlined" />
                {model.label}
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {selectedModelData && (
        <Box sx={{ mt: 2, p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
          <Typography variant="body2" color="text.secondary">
            {selectedModelData.description}
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default ModelSelector;
