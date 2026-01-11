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
- Cloud-based inference API
- Kubernetes autoscaling for multiple camera feeds
- Dashboard showing species counts and trends
