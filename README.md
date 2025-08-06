# TranAD Time Series Anomaly Detection

Personal implementation of TranAD (Transformer-based Anomaly Detection) for time series forecasting and anomaly detection.

## Overview
This repository contains a customized version of TranAD adapted for various time series datasets including SMD, SWaT, NAB, and others. The implementation focuses on detecting anomalies in multivariate time series data using transformer-based architectures.

## Key Features
- Multi-dataset support (SMD, SWaT, NAB, SMAP_MSL, etc.)
- Transformer-based anomaly detection
- Preprocessing utilities for time series data
- Visualization and plotting tools
- Configurable model parameters

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Datasets
The repository includes support for multiple benchmark datasets in the `data/` directory, each with their respective preprocessing and evaluation scripts.