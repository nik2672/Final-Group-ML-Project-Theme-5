import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Card,
  CardContent,
  CardMedia,
  Chip,
  Alert,
} from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import DownloadIcon from '@mui/icons-material/Download';
import CloseIcon from '@mui/icons-material/Close';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';

function OutputViewer({ outputFiles, model }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageErrors, setImageErrors] = useState(new Set());

  if (!outputFiles || outputFiles.length === 0) {
    return null;
  }

  const imageFiles = outputFiles.filter(file => file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg'));
  const csvFiles = outputFiles.filter(file => file.endsWith('.csv'));

  const handleImageClick = (filename) => {
    setSelectedImage(filename);
  };

  const handleDownload = async (filename) => {
    try {
      const response = await fetch(`/api/results/${filename}`);
      if (!response.ok) {
        alert(`File not found: ${filename}`);
        return;
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert(`Failed to download ${filename}`);
    }
  };

  const handleOpenResultsFolder = () => {
    // Try to open the results folder using the file API
    const folderPath = model && ['kmeans', 'dbscan', 'birch', 'optics', 'hdbscan'].includes(model) 
      ? 'results/clustering/' 
      : 'results/forecasting/';
    
    // Create a more helpful message
    const message = `Results are saved in: ${folderPath}\n\nYou can find your files at:\nC:\\Users\\13min\\Final-Group-ML-Project-Theme-5\\${folderPath}`;
    alert(message);
  };

  const handleImageError = (filename) => {
    setImageErrors(prev => new Set([...prev, filename]));
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
                <Card elevation={3}>
                  <Box sx={{ position: 'relative' }}>
                    {!imageErrors.has(file) ? (
                      <CardMedia
                        component="img"
                        height="200"
                        image={`/api/results/${file}`}
                        alt={file}
                        sx={{ 
                          objectFit: 'contain',
                          bgcolor: 'grey.50',
                          cursor: 'pointer',
                        }}
                        onClick={() => handleImageClick(file)}
                        onError={() => handleImageError(file)}
                      />
                    ) : (
                      <Box
                        sx={{
                          height: 200,
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: 'grey.100',
                          cursor: 'pointer',
                        }}
                        onClick={() => handleImageClick(file)}
                      >
                        <ImageIcon sx={{ fontSize: 48, color: 'grey.400', mb: 1 }} />
                        <Typography variant="caption" color="text.secondary">
                          Image not available
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Click to try loading
                        </Typography>
                      </Box>
                    )}
                    <IconButton
                      sx={{
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        bgcolor: 'rgba(255, 255, 255, 0.8)',
                        '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.9)' },
                      }}
                      size="small"
                      onClick={() => handleImageClick(file)}
                    >
                      <ZoomInIcon />
                    </IconButton>
                  </Box>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Typography variant="body2" noWrap sx={{ flexGrow: 1, mr: 1 }}>
                        {file}
                      </Typography>
                      <Chip label="PNG" size="small" variant="outlined" color="primary" />
                    </Box>
                  </CardContent>
                </Card>
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
                <Card elevation={2}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      <InsertDriveFileIcon color="primary" />
                      <Typography variant="body2" noWrap sx={{ flexGrow: 1 }}>
                        {file}
                      </Typography>
                      <Chip label="CSV" size="small" variant="outlined" color="secondary" />
                    </Box>
                    <Button
                      fullWidth
                      variant="contained"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleDownload(file)}
                      size="small"
                    >
                      Download
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Image Preview Dialog */}
      <Dialog
        open={Boolean(selectedImage)}
        onClose={() => setSelectedImage(null)}
        maxWidth="xl"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <ImageIcon color="primary" />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              {selectedImage}
            </Typography>
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={() => selectedImage && handleDownload(selectedImage)}
              size="small"
            >
              Download
            </Button>
            <IconButton onClick={() => setSelectedImage(null)}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent sx={{ textAlign: 'center', p: 3 }}>
          {selectedImage && (
            <Box>
              <img
                src={`/api/results/${selectedImage}`}
                alt={selectedImage}
                style={{ 
                  width: '100%', 
                  height: 'auto', 
                  maxHeight: '70vh',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                }}
                onError={() => (
                  <Alert severity="warning" sx={{ mt: 2 }}>
                    Unable to load image: {selectedImage}
                  </Alert>
                )}
              />
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}

export default OutputViewer;
