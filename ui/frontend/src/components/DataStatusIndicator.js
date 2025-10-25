import React, { useState, useEffect } from 'react';
import {
  Box,
  Alert,
  AlertTitle,
  CircularProgress,
  Chip,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import axios from 'axios';

function DataStatusIndicator() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkDataStatus();
  }, []);

  const checkDataStatus = async () => {
    try {
      const response = await axios.get('/api/data-status');
      setStatus(response.data);
    } catch (error) {
      console.error('Failed to check data status:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <CircularProgress size={20} />
        <span>Checking data files...</span>
      </Box>
    );
  }

  if (!status) return null;

  const clusteringReady = status.clustering_train_data && status.clustering_test_data;
  const forecastingReady = status.forecasting_train_data && status.forecasting_test_data;
  const allReady = clusteringReady && forecastingReady;

  return (
    <Box sx={{ mb: 3 }}>
      {allReady ? (
        <Alert severity="success" icon={<CheckCircleIcon />}>
          <AlertTitle>Data Files Ready</AlertTitle>
          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
            <Chip label="Clustering (Train/Test) ✓" size="small" color="success" variant="outlined" />
            <Chip label="Forecasting (Train/Test) ✓" size="small" color="success" variant="outlined" />
          </Box>
        </Alert>
      ) : (
        <Alert severity="warning" icon={<ErrorIcon />}>
          <AlertTitle>Missing Data Files</AlertTitle>
          Please run feature engineering first: <code>python src/features/leakage_safe_feature_engineering.py</code>
          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
            {!clusteringReady && <Chip label="Clustering (Train/Test) ✗" size="small" color="error" variant="outlined" />}
            {!forecastingReady && <Chip label="Forecasting (Train/Test) ✗" size="small" color="error" variant="outlined" />}
          </Box>
        </Alert>
      )}
    </Box>
  );
}

export default DataStatusIndicator;
