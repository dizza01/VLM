# MICCAI 2025 VLM3D Challenge Project

This project is for Task 1 of the Vision-Language Modeling in 3D Medical Imaging (VLM3D) Challenge at MICCAI 2025.

## Project Structure
- `src/`: Source code for data loading, preprocessing, modeling, training, and inference
- `data/`: Downloaded and processed data
- `notebooks/`: Jupyter notebooks for exploration
- `docker/`: Docker-related files for submission
- `requirements.txt`: Python dependencies

## Getting Started
1. Create and activate a Python virtual environment:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Login to Hugging Face to access the dataset:
   ```sh
   huggingface-cli login
   ```
4. Download the dataset using the provided scripts.

## Challenge Details
- [MICCAI 2025 VLM3D Challenge](https://miccai2025.org/)
- Dataset: [CT-RATE on Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

## Submission
Submission is via Docker container. See `docker/` for template and instructions.

---
