import pandas as pd
import openpyxl
def import_databases() :
    we_green=pd.read_excel("../data/data_wegreen.xlsx")
    chatgpt=pd.read_excel("../data/Chatgpt_data.xlsx")
    sector=pd.read_excel("../data/wikipedia_secteur.xlsx")
    ticker_esg=pd.read_excel("../data/ticker_esg_data.xlsx")
def annual_return_volatility(ticker, year):
    start_date = f"{str(year)}-01-01"
    end_date = f"{str(year)}-12-31"

    # Récupérer les données historiques pour le ticker et la période spécifiée
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculer les rendements journaliers
    daily_returns = data['Adj Close'].pct_change().dropna()

    # Calculer le rendement annuel
    annual_return = (1 + daily_returns).prod() - 1

    # Calculer la volatilité annuelle
    annual_volatility = daily_returns.std() * np.sqrt(252)  # Il y a généralement 252 jours de trading dans une année

    return annual_return, annual_volatility
def main() :
    we_green = pd.read_excel("../data/data_wegreen.xlsx")
    chatgpt = pd.read_excel("../data/Chatgpt_data.xlsx")
    sector = pd.read_excel("../data/wikipedia_secteur.xlsx")
    ticker_esg = pd.read_excel("../data/ticker_esg_data.xlsx")
    data_carbon = pd.merge(we_green, chatgpt, left_on=[we_green.columns[0], we_green.columns[1]],
                           right_on=[chatgpt.columns[0], chatgpt.columns[1]], how='outer')
    data_carbon_esg = pd.merge(data_carbon, ticker_esg, left_on=data_carbon.columns[0], right_on=ticker_esg.columns[0],
                               how='outer')
    data_carbon_esg['Year'] = data_carbon_esg['Year'].apply(lambda x: str(x)[:4])
    data_carbon_esg['Ticker'] = data_carbon_esg['Ticker'].astype(str)
    data_carbon_esg[['annual return', 'annual volatility']] = data_carbon_esg.apply(
        lambda row: annual_return_volatility(row['Ticker'], row['Year']), axis=1, result_type='expand')
    data_carbon_esg_secteur = pd.merge(data_carbon_esg, wiki_secteur, left_on=data_carbon_esg.columns[0],
                                       right_on=wiki_secteur.columns[0], how='outer')
    data_carbon_esg_secteur
    if __name__ == "__main__":
        main()