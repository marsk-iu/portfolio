import cld3 #pycld3 zur Spracherkennung
import pandas as pd

#pandas DataFrame erstellen mit csv als Eingabe
data = pd.read_csv('subset.csv') 

#Spalte mit Sprache hinzufügen
data['Sprache'] = "" 

#Sprache für jede Bewertung erkennen und entsprechend eintragen
for i in data.index: 
    data['Sprache'].iloc[i] = cld3.get_language(data.iloc[i]['Bewertungen']).language

#neues Dataframe erstellen, welches nur jenen Bewertungen enthält, die als Deutsch erkannt wurden    
corpus = data[data['Sprache']=="de"]['Bewertungen']

#zur Kontrolle
print(data)
print(corpus)