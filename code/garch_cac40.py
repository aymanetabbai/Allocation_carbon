import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def data_import(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['returns'] = data['Close'].pct_change().fillna(method='ffill')
    return 100*data['returns'].dropna()
def pacf_plot(returns):
    plot_pacf(np.array(returns) ** 2)
def fit_garch_model(returns, p=1, q=1):
    model = arch_model(returns, vol='Garch', p=p, q=q, dist='Normal')
    results = model.fit()
    return results
def forecast_garch(results, horizon=30):
    forecast = results.forecast(start=0, horizon=horizon)
    return forecast


def plot_forecast(test_returns, forecast_variance):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Tracez la prévision de la variance GARCH
    ax.plot(test_returns.index, forecast_variance, label="Prévision de la variance GARCH")

    # Tracez la véritable variance(approxx)
    true_variance = (test_returns ** 2)
    ax.plot(true_variance.index, true_variance, label="Véritable variance", alpha=0.6)

    ax.set_title("Prévision GARCH de 30 jours et véritable variance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Variance")
    ax.legend()
    plt.show()

    # Créez un DataFrame avec les valeurs précises de la véritable variance et de la prévision de la variance GARCH
    variance_comparison = pd.DataFrame(
        {"Véritable variance": true_variance, "Prévision de la variance GARCH": forecast_variance})
    print(variance_comparison)


def find_best_garch(returns, max_p=5, max_q=5):
    best_aic = np.inf
    best_order = None
    best_results = None

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                model = arch_model(returns, vol='Garch', p=p, q=q, dist='Normal')
                results = model.fit(disp='off')
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, q)
                    best_results = results
            except:
                continue

    return best_order, best_results


def main():
    ticker = "^FCHI"
    start_date = "2020-01-01"
    end_date = "2021-09-30"
    test_start_date = "2021-10-01"
    test_end_date = "2021-10-31"

    # Importer les données et séparer les ensembles d'apprentissage et de test
    returns = data_import(ticker, start_date, end_date)
    test_returns = data_import(ticker, test_start_date, test_end_date)

    pacf_plot(returns)
    results = fit_garch_model(returns, 3, 3)
    print(results.summary())

    # Générer les prévisions de variance pour les 20 jours suivant l'ensemble d'apprentissage
    forecast = forecast_garch(results, horizon=20)
    forecast_variance = forecast.variance.iloc[-1].values

    plot_forecast(test_returns, forecast_variance)


if __name__ == "__main__":
    main()