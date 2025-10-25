import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';

function RunHistory({ history, onClear, onCompare }) {
  if (!history || history.length === 0) {
    return null;
  }

  const getModelColor = (model) => {
    const colors = {
      kmeans: 'primary',
      dbscan: 'secondary',
      birch: 'info',
      optics: 'warning',
      hdbscan: 'error',
      xgboost: 'success',
      arima: 'warning',
      sarima: 'info',
      lstm: 'secondary',
      gru: 'primary',
    };
    return colors[model] || 'default';
  };

  const formatMetric = (value) => {
    if (value === null || value === undefined) return 'N/A';
    return typeof value === 'number' ? value.toFixed(3) : value;
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Run History</Typography>
        <Box>
          {history.length >= 2 && (
            <Tooltip title="Compare runs">
              <IconButton onClick={onCompare} color="primary" size="small">
                <CompareArrowsIcon />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Clear history">
            <IconButton onClick={onClear} color="error" size="small">
              <DeleteIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <TableContainer component={Paper} elevation={2}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Time</TableCell>
              <TableCell>Model</TableCell>
              <TableCell>Target</TableCell>
              <TableCell>MAE / Silhouette</TableCell>
              <TableCell>RMSE / DB Index</TableCell>
              <TableCell>R² / Clusters</TableCell>
              <TableCell>Calinski-Harabasz</TableCell>
              <TableCell align="right">Time (s)</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {history.map((run, index) => {
              // Support both old format (metrics) and new format (train_metrics/test_metrics)
              const metrics = run.train_metrics || run.metrics;
              
              return (
                <TableRow key={index} hover>
                  <TableCell>{run.timestamp}</TableCell>
                  <TableCell>
                    <Chip label={run.model.toUpperCase()} size="small" color={getModelColor(run.model)} />
                  </TableCell>
                  <TableCell>{run.target_metric || 'N/A'}</TableCell>
                  <TableCell>
                    {formatMetric(metrics?.mae || metrics?.silhouette_score)}
                  </TableCell>
                  <TableCell>
                    {formatMetric(metrics?.rmse || metrics?.davies_bouldin_score)}
                  </TableCell>
                  <TableCell>
                    {formatMetric(metrics?.r2 || metrics?.n_clusters)}
                  </TableCell>
                  <TableCell>
                    {formatMetric(metrics?.calinski_harabasz_score)}
                  </TableCell>
                  <TableCell align="right">{run.execution_time.toFixed(2)}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default RunHistory;
