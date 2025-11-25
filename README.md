# Time Series Analysis Lectures

> The repository contains some slides and materials related to **time series analysis**

The code in this repository is mainly based on the [`darts`](https://unit8co.github.io/darts/) library, a Python library for easy manipulation and forecasting of time series.

[![Darts Logo](https://unit8co.github.io/darts/_images/darts-logo-light.png)](https://unit8co.github.io/darts/index.html)

The list of the method implemented in the `darts` library can be found in the [official documentation](hhttps://unit8co.github.io/darts/#forecasting-models) for forecasting and [here](https://unit8co.github.io/darts/generated_api/darts.ad.html) for anomaly detection.

To better understand the idea of multivariate time series in Darts and covariates, please refer to [this example](https://unit8co.github.io/darts/examples/01-multi-time-series-and-covariates.html) in the documentation or to the examples provided in the `examples/` folder of this repository.

## Main Handbook Contents

The `handbook.ipynb` notebook contains several example of time series analysis lectures, including:

- Overview
- Setup & Data Preparation
- Dataset Decomposition
- Forecasting Methods
- Anomaly Detection Methods

👉 [*Open the notebook on **Colab***.](https://colab.research.google.com/github/jacksalici/tsa-lectures/blob/main/handbook.ipynb)

## More Advanced Examples

- `examples/forecast_gridsearch.py`: Example of performing grid-search hyperparameter tuning for forecasting models using the `darts` library. More about hyperparameter tuning can be found in the [official documentation](https://unit8co.github.io/darts/examples/17-hyperparameter-optimization.html).
- `examples/anomaly_score.py`: Example of computing anomaly scores using the error between actual and forecasted values from a forecasting model. More about anomaly detection can be found in the [official documentation](https://unit8co.github.io/darts/examples/22-anomaly-detection-examples.html).
- `examples/forecast_multivariate_synthetic.py`: Multivariate time series forecasting example using a synthetic dataset with historical backtesting.
- `examples/forecast_multivariate_etth.py`: Multivariate time series forecasting example using the ETTh1 dataset (Electricity Transformer Temperature) with historical backtesting.

## Slides

The slides used in the lecture of November 20th, 2025 for the course of "Internet of Things" can be found in the `slides/` folder. 

## Miscellaneous 

- `misc/ma.ipynb`:  Interactive plots of **Moving Average** Smoothing effect on a random time series ([*open on **Colab***](https://colab.research.google.com/github/jacksalici/tsa-lectures/blob/main/misc/ma.ipynb))
- `misc/ema.ipynb`: Interactive plots of **Exponential Moving Average** on an impulse signal ([*open on **Colab***](https://colab.research.google.com/github/jacksalici/tsa-lectures/blob/main/misc/ema.ipynb))
