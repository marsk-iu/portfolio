import cld3 #pycld3 zur Spracherkennung
import pandas as pd

#pandas DataFrame erstellen mit csv als Eingabe
data = pd.read_csv('Bewertungen_nobreaks.csv') 

#Spalte mit Sprache hinzufügen
data['Sprache'] = "" 

#Sprache für jede Bewertung erkennen und entsprechend eintragen
for i in data.index: 
    data.loc[i, 'Sprache'] = cld3.get_language(data.iloc[i]['Bewertungen']).language
    
#neues Dataframe erstellen, welches nur jenen Bewertungen enthält, die als Deutsch erkannt wurden    
corpus = data[data['Sprache']=="de"]['Bewertungen']

#Dataframe als .csv speichern
corpus.to_csv('Bewertungen_de.csv', sep=',', index=False)

#zur Kontrolle
#print(data)
#print(corpus)