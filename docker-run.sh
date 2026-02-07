#!/bin/bash
# Run script for NAVIS container

set -e

echo "Starting NAVIS container..."

# Check if container already exists
if docker ps -a | grep -q navis; then
    echo "Removing existing container..."
    docker rm -f navis
fi

# Run container
docker run -d \
  --name navis \
  -p 8000:8000 \
  navis:latest

echo "Container started!"
echo ""
echo "Access points:"
echo "  Frontend: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/health"
echo ""
echo "To view logs:"
echo "  docker logs -f navis"
echo ""
echo "To stop:"
echo "  docker stop navis"

