# Multi-stage build for NAVIS container
# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder

# Install build dependencies
RUN apk add --no-cache python3 make g++

WORKDIR /build

# Copy frontend files
COPY vessel-detection-gui/package*.json ./
COPY vessel-detection-gui/vite.config.js ./
COPY vessel-detection-gui/index.html ./
COPY vessel-detection-gui/src ./src/
COPY vessel-detection-gui/public ./public/

# Install dependencies and build
RUN npm ci && \
    npm run build

# Stage 2: Python runtime with frontend
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements_vessel_detection.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_vessel_detection.txt

# Copy backend code
COPY motion_core.py .
COPY cv_agent.py .
COPY orchestrator.py .
COPY schemas.py .
COPY vessel_detection_server.py .
COPY c3_rest_wrapper.py .

# Copy YOLO model
COPY models/ ./models/

# Copy media directory (optional, for demo videos)
COPY media/ ./media/

# Copy built frontend from stage 1
COPY --from=frontend-builder /build/dist ./static

# Create non-root user for security
RUN useradd -m -u 1000 navis && \
    chown -R navis:navis /app

USER navis

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["python", "-m", "uvicorn", "vessel_detection_server:app", "--host", "0.0.0.0", "--port", "8000"]

