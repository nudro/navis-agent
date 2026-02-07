#!/bin/bash
# Build script for NAVIS container

set -e

echo "Building NAVIS container..."

# Build Docker image
docker build -t navis:latest .

echo "Build complete!"
echo ""
echo "To save for USB transfer:"
echo "  docker save navis:latest -o navis-container.tar"
echo "  gzip navis-container.tar"
echo ""
echo "To run:"
echo "  docker run -d --name navis -p 8000:8000 navis:latest"

