# Time Series Analysis Lectures

> The repository contains some materials related to **time series analysis**. The `handbook.ipynb` notebook includes some snippets to showcase the [`darts`](https://unit8co.github.io/darts/index.html) library applied to time series forecasting and anomaly detection tasks.

## Main Handbook Contents

- Overview
- Setup & Data Preparation
- Dataset Decomposition
- Forecasting Methods
- Anomaly Detection Methods

👉 [*Open the notebook on **Colab***.](https://colab.research.google.com/github/jacksalici/tsa-lectures/blob/main/handbook.ipynb)

### Miscellaneous 

- `misc/ma.ipynb`:  Interactive plots of **Moving Average** Smoothing effect on a random time series ([*open on **Colab***](https://colab.research.google.com/github/jacksalici/tsa-lectures/blob/main/misc/ma.ipynb))
- `misc/ema.ipynb`: Interactive plots of **Exponential Moving Average** on an impulse signal ([*open on **Colab***](https://colab.research.google.com/github/jacksalici/tsa-lectures/blob/main/misc/ema.ipynb))

## More Advanced Examples
- `examples/forecast_gridsearch.py`: Example of performing grid-search hyperparameter tuning for forecasting models using the `darts` library. More about hyperparameter tuning can be found in the [official documentation](https://unit8co.github.io/darts/examples/17-hyperparameter-optimization.html).
- `examples/anomaly_score.py`: Example of computing anomaly scores using the error between actual and forecasted values from a forecasting model. More about anomaly detection can be found in the [official documentation](https://unit8co.github.io/darts/examples/22-anomaly-detection-examples.html).
