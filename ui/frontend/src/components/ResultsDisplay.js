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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ErrorIcon from '@mui/icons-material/Error';
import OutputViewer from './OutputViewer';

function ResultsDisplay({ results, model }) {
  if (!results) return null;

  const renderClusteringResults = () => {
    // Check if we have train_metrics (new format) or metrics (old format)
    const trainMetrics = results.train_metrics || results.metrics;
    const testMetrics = results.test_metrics;
    
    if (!trainMetrics) return null;

    const renderMetricsGrid = (metrics, title) => (
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom color="text.secondary">
          {title}
        </Typography>
        <Grid container spacing={3}>
          {metrics.silhouette_score !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Silhouette Score
                  </Typography>
                  <Typography variant="h5">
                    {metrics.silhouette_score.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Higher is better (0-1)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {metrics.davies_bouldin_score !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Davies-Bouldin
                  </Typography>
                  <Typography variant="h5">
                    {metrics.davies_bouldin_score.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Lower is better
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {metrics.calinski_harabasz_score !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Calinski-Harabasz
                  </Typography>
                  <Typography variant="h5">
                    {metrics.calinski_harabasz_score.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Higher is better
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {metrics.n_clusters !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Clusters Found
                  </Typography>
                  <Typography variant="h5">
                    {metrics.n_clusters}
                  </Typography>
                  {metrics.n_outliers !== undefined && (
                    <Typography variant="caption" color="text.secondary">
                      {metrics.n_outliers} outliers
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Box>
    );

    return (
      <Box>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckCircleIcon color="success" />
          Clustering Results
        </Typography>

        {renderMetricsGrid(trainMetrics, "Training Data Performance")}
        
        {testMetrics && (
          <>
            <Divider sx={{ my: 3 }} />
            {renderMetricsGrid(testMetrics, "Test Data Performance")}
            
            {/* Show generalization alert if test performance is significantly lower */}
            {trainMetrics.silhouette_score && testMetrics.silhouette_score && 
             testMetrics.silhouette_score < trainMetrics.silhouette_score * 0.5 && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                Model may be overfitting - test performance is significantly lower than training performance.
              </Alert>
            )}
          </>
        )}

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
          {['kmeans', 'dbscan', 'birch', 'optics', 'hdbscan'].includes(model) ? renderClusteringResults() : renderForecastingResults()}

          {results.target_metric && (
            <Box sx={{ mt: 3 }}>
              <Alert severity="info" variant="outlined">
                <strong>Target Metric:</strong> {results.target_metric}
              </Alert>
            </Box>
          )}

          <OutputViewer outputFiles={results.output_files} model={model} />

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
          <Alert severity="error" icon={<ErrorIcon />} sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              {results.error_type || 'Error'}
            </Typography>
            <Typography variant="body1">
              {results.message || 'An error occurred while running the model'}
            </Typography>
          </Alert>

          {results.stack_trace && (
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Stack Trace (for debugging)</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box
                  component="pre"
                  sx={{
                    bgcolor: 'grey.900',
                    color: 'grey.100',
                    p: 2,
                    borderRadius: 1,
                    overflow: 'auto',
                    fontSize: '0.875rem',
                    fontFamily: 'monospace',
                  }}
                >
                  {results.stack_trace}
                </Box>
              </AccordionDetails>
            </Accordion>
          )}
        </Box>
      )}
    </Paper>
  );
}

export default ResultsDisplay;
