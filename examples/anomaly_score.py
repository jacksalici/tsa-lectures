from darts import TimeSeries
from darts.datasets import WeatherDataset
from darts.models import SKLearnModel
from sklearn.linear_model import Ridge
from darts.ad import ForecastingAnomalyModel, NormScorer, WassersteinScorer
from darts.ad.detectors import QuantileDetector
from darts.ad.utils import show_anomalies_from_scores
from darts.metrics import mae, rmse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # Load dataset and extract target variable
    series = WeatherDataset().load()
    target = series['p (mbar)'][:100]

    # Configure and train model
    model = SKLearnModel(model=Ridge(alpha=1.0), lags=12, output_chunk_length=6)
    model.fit(target)

    # Setup anomaly detection
    half_day = 24
    anomaly_model = ForecastingAnomalyModel(
        model=model,
        scorer=[NormScorer(ord=2), WassersteinScorer(window=half_day, window_agg=False)]
    )

    # Fit and score
    anomaly_model.fit(target, start=0.1, allow_model_training=False, verbose=True)
    anomaly_scores, forecast = anomaly_model.score(
        target, start=0.1, return_model_prediction=True, verbose=True
    )

    print("Anomaly detection completed.")

    pred_start = forecast.start_time()
    target_aligned = target[pred_start:]
    print(f"MAE: {mae(forecast, target_aligned):.3f}, RMSE: {rmse(forecast, target_aligned):.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Plot 1: Original series and forecast
    target.plot(label="Target", ax=axes[0])
    forecast.plot(label="Forecast", ax=axes[0])
    axes[0].legend()
    axes[0].set_title("Target vs Forecast")
    
    # Plot 2-3: Anomaly scores and detections
    for i, score in enumerate(anomaly_scores):
        ax = axes[1 + i]
        score.plot(label=f"{type(anomaly_model.scorers[i]).__name__}", ax=ax)
        
        # Add detector on each score
        detector = QuantileDetector(high_quantile=0.95)
        anomaly_pred = detector.fit_detect(score)
        anomaly_pred.plot(label="Detected", ax=ax, color="red", alpha=0.7)
        ax.legend()
        ax.set_title(f"Anomaly Score: {type(anomaly_model.scorers[i]).__name__}")
    
    plt.tight_layout()
    plt.show()
    
    # Combined visualization
    scorer_names = [type(s).__name__ for s in anomaly_model.scorers]
    show_anomalies_from_scores(
        series=target,
        pred_scores=anomaly_scores,
        pred_series=forecast,
        window=[1, half_day],
        title="Anomaly Detection Results",
        names_of_scorers=scorer_names
    )