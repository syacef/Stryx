# Wildlife Species Recognition using Self-Supervised Learning

### Problem
Wildlife monitoring using camera traps produces massive unlabeled image data. Manual species annotation is slow and expensive.
Build an automated wildlife recognition system using self-supervised learning to learn visual representations from unlabeled camera-trap images and videos, followed by fine-tuning for species classification.

### Dataset
- Snapshot Serengeti (too large 200G - 600G)
- Caltech Camera Traps (100G) -> prefered
- iNaturalist (200G)
- Smaller Datasets ?

### Methodology
- SSL pretraining using:
  * Contrastive learning (SimCLR / BYOL / DINO)
  * Temporal consistency from burst images
  * Fine-tuning for species classification or embedding-based retrieval

### Deployment
- **NEW**: FastAPI + PyTorch microservices architecture
- Two-service design:
  * **Ingestion Service** (CPU): Motion detection and frame filtering
  * **Inference Service** (GPU): PyTorch model inference with batching
- Redis queue for decoupled processing
- PostgreSQL/TimescaleDB for results storage
- Kubernetes autoscaling support
- REST API for easy integration

## Quick Start

### Setup
```bash
# Run setup script
./setup.sh

# Start services
cd src
docker-compose up -d

# Check health
curl http://localhost:8000/health  # Ingestion
curl http://localhost:8001/health  # Inference
```

### Process a Video
```bash
# Copy video to data directory
cp /path/to/video.mp4 data/videos/

# Submit for processing
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"video_path": "video.mp4"}'

# Or use the example script
python3 example.py
```

### Project Structure
```
├── src/
│   ├── ingestion_service/    # CPU-optimized video processing
│   ├── inference_service/    # GPU-optimized model inference
│   ├── model/                # Model training code
│   ├── shared/               # Shared utilities
│   ├── docker-compose.yaml   # Service orchestration
│   └── README.md            # Detailed documentation
├── k8s/                      # Kubernetes manifests
├── MIGRATION.md             # Migration guide from DeepStream
├── setup.sh                 # Setup script
└── example.py               # Example usage script
```

## Documentation

- **[src/README.md](src/README.md)**: Detailed service documentation
- **[MIGRATION.md](MIGRATION.md)**: Migration from DeepStream to FastAPI
- **[src/model/README.md](src/model/README.md)**: Model training guide

## Features

✅ **Motion Detection**: Filters 60% of empty frames before GPU processing  
✅ **Efficient Batching**: Process multiple frames simultaneously on GPU  
✅ **Horizontal Scaling**: Scale ingestion and inference independently  
✅ **REST API**: Easy integration with other systems  
✅ **Real-time Monitoring**: Health checks and statistics endpoints  
✅ **PostgreSQL Storage**: Queryable results with TimescaleDB support
