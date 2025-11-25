"""
Example: Use Darts' `gridsearch` classmethod for simple (non-deep) models.

This demonstrates using the `ForecastingModel.gridsearch` classmethod
for `ExponentialSmoothing` and `ARIMA` on the `AirPassengers` dataset.
"""

from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from darts.datasets import AirPassengersDataset
from darts import TimeSeries
from darts.models import ExponentialSmoothing, ARIMA, Theta
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.metrics import mape




def train_val_split(series: TimeSeries, frac: float = 0.8):
    n = len(series)
    if n < 4:
        raise ValueError("Series too short to split")
    cutoff_idx = int(n * frac) - 1
    cutoff = series.time_index[cutoff_idx]
    return series.split_after(cutoff)


def main():
    warnings.filterwarnings("ignore")

    # Load example dataset
    series = AirPassengersDataset().load()
    train, val = train_val_split(series, frac=0.8)
    print(f"Training length: {len(train)}, Validation length: {len(val)}")

   
    print("\nGrid-search ExponentialSmoothing (using gridsearch classmethod)")

    params_es = {
        "trend": [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
        "seasonal": [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
        "seasonal_periods": [None, 12],
    }

    
    best_es_model, best_es_params, best_es_score = ExponentialSmoothing.gridsearch(
            parameters=params_es,
            series=train,
            val_series=val,
            metric=mape,
            n_jobs=1,
            verbose=False,
        )
    
    print(f"Best ExponentialSmoothing MAPE={best_es_score:.3f}, params={best_es_params}")
    

  
    print("\nGrid-search ARIMA (using gridsearch classmethod)")
    params_arima = {"p": [p for p in range(3)], "d": [d for d in range(2)], "q": [q for q in range(3)]}

    
    best_arima_model, best_arima_params, best_arima_score = ARIMA.gridsearch(
            parameters=params_arima,
            series=train,
            val_series=val,
            metric=mape,
            n_jobs=1,
            verbose=False,
        )
    print(f"Best ARIMA MAPE={best_arima_score:.3f}, params={best_arima_params}")
  


    print("\nBaseline: Theta")
    theta = Theta()
    theta.fit(train)
    pred = theta.predict(len(val))
    print(f"Theta MAPE={mape(val, pred):.3f}")
  


if __name__ == "__main__":
    main()
