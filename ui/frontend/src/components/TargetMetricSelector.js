import React from 'react';
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
} from '@mui/material';

const targetMetrics = [
  { value: 'avg_latency', label: 'Average Latency (ms)', description: 'Network response time' },
  { value: 'upload_bitrate', label: 'Upload Bitrate (Mbps)', description: 'Upload speed' },
  { value: 'download_bitrate', label: 'Download Bitrate (Mbps)', description: 'Download speed' },
];

function TargetMetricSelector({ value, onChange }) {
  const selectedMetric = targetMetrics.find(m => m.value === value);

  return (
    <Box sx={{ mb: 3 }}>
      <FormControl fullWidth variant="outlined">
        <InputLabel id="target-metric-label">Target Metric</InputLabel>
        <Select
          labelId="target-metric-label"
          id="target-metric-select"
          value={value}
          label="Target Metric"
          onChange={(e) => onChange(e.target.value)}
        >
          {targetMetrics.map((metric) => (
            <MenuItem key={metric.value} value={metric.value}>
              {metric.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {selectedMetric && (
        <Box sx={{ mt: 1, px: 1 }}>
          <Typography variant="caption" color="text.secondary">
            {selectedMetric.description}
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default TargetMetricSelector;
