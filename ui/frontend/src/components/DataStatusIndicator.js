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

  const allReady = status.clustering_data && status.forecasting_data;

  return (
    <Box sx={{ mb: 3 }}>
      {allReady ? (
        <Alert severity="success" icon={<CheckCircleIcon />}>
          <AlertTitle>Data Files Ready</AlertTitle>
          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
            <Chip label="Clustering ✓" size="small" color="success" variant="outlined" />
            <Chip label="Forecasting ✓" size="small" color="success" variant="outlined" />
          </Box>
        </Alert>
      ) : (
        <Alert severity="warning" icon={<ErrorIcon />}>
          <AlertTitle>Missing Data Files</AlertTitle>
          Please run feature engineering first: <code>python src/features/feature_engineering.py</code>
          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
            {!status.clustering_data && <Chip label="Clustering ✗" size="small" color="error" variant="outlined" />}
            {!status.forecasting_data && <Chip label="Forecasting ✗" size="small" color="error" variant="outlined" />}
          </Box>
        </Alert>
      )}
    </Box>
  );
}

export default DataStatusIndicator;
