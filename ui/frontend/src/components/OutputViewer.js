import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
} from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import DownloadIcon from '@mui/icons-material/Download';
import CloseIcon from '@mui/icons-material/Close';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';

function OutputViewer({ outputFiles, model }) {
  const [selectedImage, setSelectedImage] = useState(null);

  if (!outputFiles || outputFiles.length === 0) {
    return null;
  }

  const imageFiles = outputFiles.filter(file => file.endsWith('.png'));
  const csvFiles = outputFiles.filter(file => file.endsWith('.csv'));

  const handleImageClick = (filename) => {
    setSelectedImage(filename);
  };

  const handleDownload = (filename) => {
    window.open(`/api/results/${filename}`, '_blank');
  };

  const handleOpenResultsFolder = () => {
    alert('Results are saved in: results/clustering/ or results/forecasting/');
  };

  return (
    <Box sx={{ mt: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Output Files</Typography>
        <Button
          startIcon={<FolderOpenIcon />}
          onClick={handleOpenResultsFolder}
          size="small"
          variant="outlined"
        >
          Open Results Folder
        </Button>
      </Box>

      {imageFiles.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom color="text.secondary">
            Visualizations
          </Typography>
          <Grid container spacing={2}>
            {imageFiles.map((file, idx) => (
              <Grid item xs={12} sm={6} md={4} key={idx}>
                <Paper
                  elevation={2}
                  sx={{
                    p: 2,
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'action.hover' },
                    transition: 'background-color 0.2s',
                  }}
                  onClick={() => handleImageClick(file)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ImageIcon color="primary" />
                    <Typography variant="body2" sx={{ flexGrow: 1 }}>
                      {file}
                    </Typography>
                  </Box>
                  <Box sx={{ mt: 1, textAlign: 'center' }}>
                    <img
                      src={`/api/results/${file}`}
                      alt={file}
                      style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {csvFiles.length > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom color="text.secondary">
            Data Files
          </Typography>
          <Grid container spacing={2}>
            {csvFiles.map((file, idx) => (
              <Grid item xs={12} sm={6} md={4} key={idx}>
                <Paper
                  elevation={2}
                  sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                >
                  <Typography variant="body2">{file}</Typography>
                  <IconButton
                    size="small"
                    color="primary"
                    onClick={() => handleDownload(file)}
                  >
                    <DownloadIcon />
                  </IconButton>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Image Preview Dialog */}
      <Dialog
        open={Boolean(selectedImage)}
        onClose={() => setSelectedImage(null)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          {selectedImage}
          <IconButton
            onClick={() => setSelectedImage(null)}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {selectedImage && (
            <img
              src={`/api/results/${selectedImage}`}
              alt={selectedImage}
              style={{ width: '100%', height: 'auto' }}
            />
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}

export default OutputViewer;
