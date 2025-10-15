import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Divider,
  Chip,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

function ResultsDisplay({ results, model }) {
  if (!results) return null;

  const renderClusteringResults = () => {
    if (!results.metrics) return null;

    return (
      <Box>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckCircleIcon color="success" />
          Clustering Results
        </Typography>

        <Grid container spacing={3} sx={{ mt: 2 }}>
          {results.metrics.silhouette_score !== undefined && (
            <Grid item xs={12} sm={4}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Silhouette Score
                  </Typography>
                  <Typography variant="h4">
                    {results.metrics.silhouette_score.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Higher is better (0-1)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {results.metrics.davies_bouldin_score !== undefined && (
            <Grid item xs={12} sm={4}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Davies-Bouldin Index
                  </Typography>
                  <Typography variant="h4">
                    {results.metrics.davies_bouldin_score.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Lower is better
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {results.metrics.n_clusters !== undefined && (
            <Grid item xs={12} sm={4}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Clusters Found
                  </Typography>
                  <Typography variant="h4">
                    {results.metrics.n_clusters}
                  </Typography>
                  {results.metrics.n_outliers !== undefined && (
                    <Typography variant="caption" color="text.secondary">
                      {results.metrics.n_outliers} outliers
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>

        {results.output_files && results.output_files.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              Output Files
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {results.output_files.map((file, idx) => (
                <Chip key={idx} label={file} variant="outlined" />
              ))}
            </Box>
          </Box>
        )}
      </Box>
    );
  };

  const renderForecastingResults = () => {
    if (!results.metrics) return null;

    return (
      <Box>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckCircleIcon color="success" />
          Forecasting Results
        </Typography>

        <Grid container spacing={3} sx={{ mt: 2 }}>
          {results.metrics.mae !== undefined && (
            <Grid item xs={12} sm={4}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Mean Absolute Error
                  </Typography>
                  <Typography variant="h4">
                    {results.metrics.mae.toFixed(3)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {results.metrics.rmse !== undefined && (
            <Grid item xs={12} sm={4}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Root Mean Squared Error
                  </Typography>
                  <Typography variant="h4">
                    {results.metrics.rmse.toFixed(3)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {results.metrics.r2 !== undefined && (
            <Grid item xs={12} sm={4}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    R² Score
                  </Typography>
                  <Typography variant="h4">
                    {results.metrics.r2.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Closer to 1 is better
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>

        {results.output_files && results.output_files.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              Output Files
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {results.output_files.map((file, idx) => (
                <Chip key={idx} label={file} variant="outlined" />
              ))}
            </Box>
          </Box>
        )}
      </Box>
    );
  };

  return (
    <Paper elevation={3} sx={{ p: 4, bgcolor: 'background.paper' }}>
      {results.status === 'success' && (
        <Box>
          {model === 'kmeans' || model === 'dbscan' ? renderClusteringResults() : renderForecastingResults()}

          {results.execution_time && (
            <Box sx={{ mt: 4 }}>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="body2" color="text.secondary" align="center">
                Execution time: {results.execution_time.toFixed(2)}s
              </Typography>
            </Box>
          )}
        </Box>
      )}

      {results.status === 'error' && (
        <Box>
          <Typography variant="h6" color="error" gutterBottom>
            Error
          </Typography>
          <Typography variant="body1">
            {results.message || 'An error occurred while running the model'}
          </Typography>
        </Box>
      )}
    </Paper>
  );
}

export default ResultsDisplay;
