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
    // Support both old format (results.metrics) and new format (train_metrics/test_metrics)
    const trainMetrics = results.train_metrics || results.metrics;
    const testMetrics = results.test_metrics;

    if (!trainMetrics) return null;

    // Check for overfitting
    const hasOverfitting = testMetrics && trainMetrics.mae && testMetrics.mae > trainMetrics.mae * 1.5;

    return (
      <Box>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckCircleIcon color="success" />
          Forecasting Results
        </Typography>

        {hasOverfitting && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            ⚠️ Potential overfitting detected: Test error is significantly higher than train error
          </Alert>
        )}

        {/* Train Metrics */}
        <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
          Training Performance
        </Typography>
        <Grid container spacing={3}>
          {trainMetrics.mae !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    MAE (Train)
                  </Typography>
                  <Typography variant="h5">
                    {trainMetrics.mae.toFixed(3)}
                  </Typography>
                  {trainMetrics.n_samples && (
                    <Typography variant="caption" color="text.secondary">
                      n={trainMetrics.n_samples.toLocaleString()}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          )}

          {trainMetrics.rmse !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    RMSE (Train)
                  </Typography>
                  <Typography variant="h5">
                    {trainMetrics.rmse.toFixed(3)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {trainMetrics.r2 !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    R² Score (Train)
                  </Typography>
                  <Typography variant="h5">
                    {trainMetrics.r2.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Closer to 1 is better
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {trainMetrics.aic !== undefined && (
            <Grid item xs={12} sm={3}>
              <Card elevation={2}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    AIC (Train)
                  </Typography>
                  <Typography variant="h5">
                    {trainMetrics.aic.toFixed(1)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Lower is better
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>

        {/* Test Metrics */}
        {testMetrics && (
          <>
            <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
              Test Performance
            </Typography>
            <Grid container spacing={3}>
              {testMetrics.mae !== undefined && (
                <Grid item xs={12} sm={3}>
                  <Card elevation={2} sx={{ bgcolor: hasOverfitting ? '#fff3e0' : 'inherit' }}>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        MAE (Test)
                      </Typography>
                      <Typography variant="h5" color={hasOverfitting ? 'warning.main' : 'inherit'}>
                        {testMetrics.mae.toFixed(3)}
                      </Typography>
                      {testMetrics.n_samples && (
                        <Typography variant="caption" color="text.secondary">
                          n={testMetrics.n_samples.toLocaleString()}
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {testMetrics.rmse !== undefined && (
                <Grid item xs={12} sm={3}>
                  <Card elevation={2}>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        RMSE (Test)
                      </Typography>
                      <Typography variant="h5">
                        {testMetrics.rmse.toFixed(3)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {testMetrics.r2 !== undefined && (
                <Grid item xs={12} sm={3}>
                  <Card elevation={2}>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        R² Score (Test)
                      </Typography>
                      <Typography variant="h5">
                        {testMetrics.r2.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Closer to 1 is better
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
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
