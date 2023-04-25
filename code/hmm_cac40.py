import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
import yfinance as yf
import matplotlib.pyplot as plt

def get_return(ticker, period, interval):
    #############################
    # input: ticker and interval
    # output: return

    ticker = yf.Ticker(str(ticker))
    data = ticker.history(period=str(period), interval=str(interval))
    returns = data.Close.pct_change().dropna()

    return returns


def return_dist_stats(asset_return):
    # Input: return series
    # Output: stats and chart

    # Histogramme et estimation de la densité du noyau
    plt.figure(figsize=(10, 6))
    sns.histplot(asset_return, kde=True)

    # Calcul des statistiques supplémentaires
    median = asset_return.median()
    mode = asset_return.mode().get(0, "N/A")
    q1 = asset_return.quantile(0.25)
    q3 = asset_return.quantile(0.75)

    # Affichage des statistiques de la distribution des rendements
    print('Statistics of the return distribution')
    print('-' * 50)
    print('Length of the return series:', asset_return.shape)
    print('Mean:', asset_return.mean() * 100, '%')
    print('Standard Deviation:', asset_return.std() * 100, '%')
    print('Skew:', asset_return.skew())
    print('Kurtosis:', asset_return.kurtosis())
    print('Median:', median * 100, '%')
    print('Mode:', mode * 100, '%' if mode != "N/A" else "N/A")
    print('Q1 (25th percentile):', q1 * 100, '%')
    print('Q3 (75th percentile):', q3 * 100, '%')

    plt.show()

def test_dist(asset_return, mode, alpha=0.05):
    """
    Fonction pour tester la moyenne ou la normalité de la distribution des rendements d'un actif.

    :param asset_return: Rendements d'un actif
    :param mode: 'mean' pour tester la moyenne, 'normal' pour tester la normalité
    :param alpha: Seuil de signification en décimal (par défaut 0.01)
    :return: Affiche le résultat du test et la valeur p
    """

    # Effectuer le test en fonction du mode choisi
    if mode == 'mean':
        t_stat, p = stats.ttest_1samp(asset_return, popmean=0, alternative='two-sided')
    elif mode == 'normal':
        k2, p = stats.normaltest(asset_return)
    else:
        raise ValueError("Le mode doit être 'mean' ou 'normal'")

    def test(p, alpha):
        print(f"p = {p:.5g}")
        if p < alpha:
            print("L'hypothèse nulle peut être rejetée")
        else:
            print("L'hypothèse nulle ne peut pas être rejetée")

    return test(p, alpha)


def gaussian_hmm(returns_df, n_components):
    # creating GaussianHMM object and training
    model = GaussianHMM(n_components, covariance_type="full", n_iter=20000)
    model.fit(np.column_stack([returns_df]))

    return model
def asset_summary(rt):
    print("Mean:")
    print(np.mean(rt))

    print("Standard Deviation:")
    print(np.std(rt))

    print("Value at Risk (VaR):")
    print(np.percentile(rt, 5))

    print("VaR / 1.64:")
    print(-np.percentile(rt, 5) / stats.norm.ppf(0.05))

    print("(VaR - mean) / 1.64:")
    print((np.percentile(rt, 5) - np.mean(rt)) / stats.norm.ppf(0.05))

    print("Kolmogorov-Smirnov Test:")
    print(stats.kstest(rt, 'norm'))

    print("\n")


def plot_hidden_states(rt, trained_hmm):
    hidden_states = trained_hmm.predict(np.column_stack([rt]))

    data_normal = rt.copy(deep=True)
    data_crisis = rt.copy(deep=True)

    data_normal.iloc[np.where(hidden_states == 1)] = np.nan
    data_crisis.iloc[np.where(hidden_states == 0)] = np.nan

    plt.figure(figsize=(10, 5))
    plt.plot(data_normal, color='blue', label='Normal')
    plt.plot(data_crisis, color='orange', label='Crisis')
    plt.tight_layout()
    plt.legend()
    plt.show()




def plot_hidden_states(rt, trained_hmm):
    hidden_states = trained_hmm.predict(np.column_stack([rt]))

    data_normal = rt.copy(deep=True)
    data_crisis = rt.copy(deep=True)

    data_normal.iloc[np.where(hidden_states == 0)] = np.nan
    data_crisis.iloc[np.where(hidden_states == 1)] = np.nan

    plt.figure(figsize=(10, 5))
    plt.plot(data_normal, color='blue', label='Normal')
    plt.plot(data_crisis, color='orange', label='Crisis')
    plt.tight_layout()
    plt.legend()
    plt.show()

def main() :
    returns = get_return(ticker='^FCHI', period='max', interval='1d')
    return_dist_stats(returns)
    test_dist(returns, 'mean')
    test_dist(returns, 'normal')
    n_components = 2
    # trained hmm model
    trained_hmm = gaussian_hmm(returns, n_components)
    plot_hidden_states(returns, trained_hmm)
if __name__ == "__main__":
    main()