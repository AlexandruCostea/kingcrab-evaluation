# 🧠 KingCrab-Evaluation

This repository contains all training logic, model definitions, and data preprocessing tools for training chess position evaluation networks (NNUE, CNN, and distilled variants) to be used inside the [KingCrab engine](https://github.com/AlexandruCostea/KingCrab)

---

## Overview

This project supports training different types of neural networks to evaluate chess positions:
 **NNUE** (Efficiently Updatable Neural Network) – for high-speed inference in chess-specific format
- **CNN** – a general-purpose model using depthwise-separable convolution
- **CNN with Distillation** – a student CNN model trained using outputs from a stronger teacher model (e.g., ResNet)

The trained models can be exported to **ONNX format** and integrated into the Rust chess engine backend.

---

### 🧪 Dataset Preparation
We recommend using the [Lichess evaluation dataset](https://database.lichess.org/#evals). This repo includes a parser to parse their .zst files and filter only the relevant data for network training.
## Prerequisites
- Rust (latest stable)
- `cargo` (Rust package manager)

### Generating the processed dataset
```bash
cd src/data_preprocessor
cargo run --release <path_to_lichess_zst> <path_for_newly_processed_data>
```

### 🤖 Model Training, Evaluating and Exporting
- Select the desired script from **src/**
- Run the script, following the instructions provided in the **parse_args()** method

## ⚙️ Setup Guide

### 📦 Prerequisites
- a python environment tool **(preferably conda)**

### Create an environment

```bash
conda create --name=<env_name> python=3.12
conda activate <env_name>
```

### Install the dependencies
```bash
pip install -r requirements.txt
```
