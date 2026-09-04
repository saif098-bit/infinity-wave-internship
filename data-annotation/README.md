# Data Annotation — Cricket Computer Vision Project

Hands-on data annotation work for a real-world computer vision project at Infinity Wave Inc, focused on labeling cricket match footage to build training data for AI models.

## About the Project

The project involves annotating match footage (cricket) using **CVAT** (Computer Vision Annotation Tool) to produce high-quality labeled datasets — the kind of labeled data foundation that computer vision models are trained on.

## Week 1 — What I Worked On

| Task | Technique | Details |
|---|---|---|
| Pitch annotation | Polygon annotation | Marked the cricket pitch boundary — the reference every other annotation in the frame is measured against |
| Bowling stump annotation | Bounding box | Tight, pixel-accurate boxes around small fixed objects (stumps) |
| Bowler annotation | Object tracking | Tracked the bowler across frames while keeping labels accurate and consistent as they move |
| Batsman annotation | Object tracking | Labeled the batsman for downstream sports analytics / AI training use |

## Skills Developed

Data annotation, bounding boxes, polygon annotation, object tracking, CV fundamentals, dataset quality, attention to detail, AI data preparation, label accuracy, sports datasets.

## Key Takeaways

- Precision matters in every annotation — small errors compound across a dataset.
- Consistency across frames is what makes annotated data usable for training.
- Reliable AI models start with reliable, carefully labeled data.

## Reports

Data Annotation Report with screenshots from the CVAT workspace.

- [Intern Journey](./Intern Journey.pdf)
