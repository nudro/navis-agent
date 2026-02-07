# NAVIS Docker Container Setup for C3.ai Integration

## Overview

This document describes how to package the entire NAVIS system (backend + frontend) into a single Docker container for C3.ai integration. The container can be built offline, transferred via USB, and run locally.

## Container Architecture

### Filesystem Layout

```
/app/                          # Container working directory (UID 1000, non-root)
├── motion_core.py             # Motion calculation utilities (shared)
├── cv_agent.py                # CV Agent
├── orchestrator.py            # Orchestrator Agent
├── schemas.py                 # Pydantic schemas
├── vessel_detection_server.py # FastAPI server (main entry point)
├── c3_rest_wrapper.py         # C3.ai REST wrapper
├── models/
│   └── yolov8n_vessels.pt     # YOLO model (REQUIRED)
├── media/                     # Video files (optional, can be mounted)
│   └── *.mp4
└── static/                     # Built frontend (from Vite build)
    ├── index.html
    ├── assets/
    │   ├── *.js
    │   └── *.css
    └── ...
```

### Build Process

**Stage 1: Frontend Builder (Node.js Alpine)**
- Copies frontend source code
- Runs `npm ci` to install dependencies
- Runs `npm run build` to create production build
- Output: `dist/` folder with static assets

**Stage 2: Runtime (Python 3.10-slim)**
- Installs system dependencies (OpenCV libraries, etc.)
- Installs Python packages from `requirements_vessel_detection.txt`
- Copies all backend Python code
- Copies built frontend from Stage 1 to `/app/static`
- Creates non-root user `navis` (UID 1000)
- Exposes port 8000
- Runs FastAPI server via uvicorn

## Building the Container

### Prerequisites
- Docker installed
- All source code in place
- YOLO model at `models/yolov8n_vessels.pt`

### Build Command

```bash
# Using build script
./docker-build.sh

# OR manually
docker build -t navis:latest .
```

### Save for USB Transfer

```bash
# Save image to tar file
docker save navis:latest -o navis-container.tar

# Compress (optional, saves space)
gzip navis-container.tar

# Result: navis-container.tar.gz (can be copied to USB)
```

### Load on Target Machine

```bash
# Copy from USB to target machine, then:
docker load -i navis-container.tar.gz

# Verify
docker images | grep navis
```

## Running the Container

### Basic Run

```bash
docker run -d \
  --name navis \
  -p 8000:8000 \
  navis:latest
```

### With GPU Support

```bash
docker run -d \
  --name navis \
  --gpus all \
  -p 8000:8000 \
  navis:latest
```

### With External Video Directory

```bash
docker run -d \
  --name navis \
  -p 8000:8000 \
  -v /path/to/videos:/app/media:ro \
  navis:latest
```

### Using Run Script

```bash
./docker-run.sh
```

## API Endpoints

### Frontend (GUI)
- `GET /` - Serves React frontend
- `GET /static/*` - Static assets (JS, CSS)
- All existing WebSocket endpoints remain functional

### C3.ai REST Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "NAVIS",
  "timestamp": "2024-02-06T16:30:00.000000",
  "model_loaded": true,
  "device": "cpu"
}
```

#### `POST /infer`
Inference endpoint. Accepts video path or frame data.

**Request (Video Path):**
```json
{
  "video_path": "vid1.mp4",
  "frame_limit": 100
}
```

**Request (Frame Data):**
```json
{
  "frame_data": "base64_encoded_image_string",
  "format": "base64"
}
```

**Response:**
```json
{
  "video_path": "/app/media/vid1.mp4",
  "fps": 30,
  "resolution": [1920, 1080],
  "total_frames": 300,
  "processed_frames": 100,
  "detections": [
    {
      "frame": 0,
      "detections": [
        {
          "bbox": [100.0, 200.0, 300.0, 400.0],
          "confidence": 0.85,
          "centroid": [200.0, 300.0],
          "area": 20000.0
        }
      ],
      "detection_count": 1
    }
  ],
  "summary": {
    "total_detections": 150,
    "frames_with_detections": 80,
    "avg_detections_per_frame": 1.5
  },
  "timestamp": "2024-02-06T16:30:00.000000"
}
```

#### `GET /metrics`
System metrics endpoint.

**Response:**
```json
{
  "timestamp": "2024-02-06T16:30:00.000000",
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 62.5,
    "memory_available_mb": 4096
  },
  "model": {
    "device": "cpu",
    "model_path": "models/yolov8n_vessels.pt",
    "model_loaded": true
  },
  "gpu": {
    "available": false
  }
}
```

## C3.ai Integration

### From C3.ai (TypeScript/JavaScript)

```typescript
// Health check
const health = await c3.http.get('http://navis-container:8000/health');

// Inference
const response = await c3.http.post('http://navis-container:8000/infer', {
  video_path: 'vid1.mp4',
  frame_limit: 100
});

const detections = response.data.detections;
const summary = response.data.summary;
```

### From C3.ai (Python)

```python
import requests

# Health check
health = requests.get('http://navis-container:8000/health').json()

# Inference
response = requests.post('http://navis-container:8000/infer', json={
    'video_path': 'vid1.mp4',
    'frame_limit': 100
})
result = response.json()
```

## Example curl Commands

### Health Check
```bash
curl http://localhost:8000/health
```

### Inference (Video Path)
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "vid1.mp4",
    "frame_limit": 10
  }'
```

### Inference (Frame Data)
```bash
# Encode image to base64 first
FRAME_B64=$(base64 -w 0 frame.jpg)

curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"frame_data\": \"$FRAME_B64\",
    \"format\": \"base64\"
  }"
```

### Metrics
```bash
curl http://localhost:8000/metrics
```

## Access Points

- **Frontend GUI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Inference**: http://localhost:8000/infer
- **Metrics**: http://localhost:8000/metrics

## Security

1. **Non-root user**: Container runs as `navis` user (UID 1000)
2. **Minimal ports**: Only port 8000 exposed
3. **Read-only volumes**: Media directory can be mounted as read-only
4. **No secrets in image**: All configuration via environment or volumes
5. **Health checks**: Built-in health monitoring

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs navis

# Check if port is in use
netstat -tuln | grep 8000
```

### Model not found
- Ensure `models/yolov8n_vessels.pt` exists in container
- Check model path in logs: `docker logs navis`

### Frontend not loading
- Verify `static/` directory exists after build
- Check that Vite build completed successfully
- View container files: `docker exec navis ls -la /app/static`

### GPU not available
- Install nvidia-docker2 on host
- Use `--gpus all` flag
- Verify CUDA: `docker exec navis python -c "import torch; print(torch.cuda.is_available())"`

## File Structure Summary

```
maritime_agent/
├── Dockerfile                    # Multi-stage build
├── .dockerignore                # Build exclusions
├── docker-build.sh              # Build script
├── docker-run.sh                # Run script
├── c3_rest_wrapper.py           # C3.ai REST wrapper
├── CONTAINER_ARCHITECTURE.md    # Detailed architecture
├── C3_INTEGRATION_EXAMPLES.md   # Integration examples
└── [existing code files...]
```

