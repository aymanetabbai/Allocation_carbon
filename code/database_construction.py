import pandas as pd
import openpyxl

we_green=pd.read_excel("../data/data_wegreen.xlsx")
chatgpt=pd.read_excel("../data/Chatgpt_data.xlsx")
sector=pd.read_excel("../data/wikipedia_secteur.xlsx")
data_carbon = pd.merge(we_green, chatgpt, left_on=[we_green.columns[0],we_green.columns[1]], right_on=[chatgpt.columns[0],chatgpt.columns[1]], how='outer')
