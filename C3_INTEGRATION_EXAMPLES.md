# C3.ai Integration Examples

## Container Setup

### Build Container
```bash
./docker-build.sh
# OR
docker build -t navis:latest .
```

### Save for USB Transfer
```bash
docker save navis:latest -o navis-container.tar
gzip navis-container.tar
```

### Load and Run on Target Machine
```bash
# Load from USB
docker load -i navis-container.tar.gz

# Run container
docker run -d \
  --name navis \
  -p 8000:8000 \
  navis:latest
```

## REST API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

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

**curl:**
```bash
curl http://localhost:8000/health
```

### 2. Inference (Video Path)

**Endpoint:** `POST /infer`

**Request:**
```json
{
  "video_path": "vid1.mp4",
  "frame_limit": 100
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

**curl:**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "vid1.mp4",
    "frame_limit": 10
  }'
```

### 3. Inference (Frame Data)

**Endpoint:** `POST /infer`

**Request:**
```json
{
  "frame_data": "base64_encoded_image_string",
  "format": "base64"
}
```

**Response:**
```json
{
  "detections": [
    {
      "bbox": [100.0, 200.0, 300.0, 400.0],
      "confidence": 0.85,
      "centroid": [200.0, 300.0],
      "area": 20000.0
    }
  ],
  "detection_count": 1,
  "frame_shape": [1080, 1920, 3],
  "timestamp": "2024-02-06T16:30:00.000000"
}
```

**curl:**
```bash
# Encode image to base64
FRAME_B64=$(base64 -w 0 frame.jpg)

curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"frame_data\": \"$FRAME_B64\",
    \"format\": \"base64\"
  }"
```

### 4. Metrics

**Endpoint:** `GET /metrics`

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

**curl:**
```bash
curl http://localhost:8000/metrics
```

## C3.ai Integration Patterns

### TypeScript/JavaScript (C3.ai)

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

// Process results
for (const frameData of detections) {
  console.log(`Frame ${frameData.frame}: ${frameData.detection_count} vessels`);
  for (const det of frameData.detections) {
    console.log(`  Vessel at ${det.centroid} with confidence ${det.confidence}`);
  }
}
```

### Python (C3.ai)

```python
import requests

# Health check
health = requests.get('http://navis-container:8000/health').json()
print(f"Status: {health['status']}")

# Inference
response = requests.post('http://navis-container:8000/infer', json={
    'video_path': 'vid1.mp4',
    'frame_limit': 100
})

result = response.json()
print(f"Processed {result['processed_frames']} frames")
print(f"Total detections: {result['summary']['total_detections']}")

# Process detections
for frame_data in result['detections']:
    print(f"Frame {frame_data['frame']}: {frame_data['detection_count']} vessels")
    for det in frame_data['detections']:
        print(f"  Bbox: {det['bbox']}, Confidence: {det['confidence']:.2f}")
```

### Batch Processing

```bash
# Process multiple videos
for video in vid1.mp4 vid2.mp4 vid3.mp4; do
  curl -X POST http://localhost:8000/infer \
    -H "Content-Type: application/json" \
    -d "{\"video_path\": \"$video\", \"frame_limit\": 50}" \
    > "results_${video}.json"
done
```

## Access Points

- **Frontend GUI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Inference**: http://localhost:8000/infer
- **Metrics**: http://localhost:8000/metrics

