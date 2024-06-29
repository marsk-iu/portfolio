from keybert import KeyBERT
import pandas as pd

#Corpus aus dem csv laden, wobei jede Zeile ein Dokument ist
corpus = pd.read_csv('fokussierte_Bewertungen.csv')

#Kontrolle, ob Zeilen des .csv mit Zeilen im Dataframe übereinstimmen
print('Number of Rows in DataFrame: ', len(corpus.index))

kw_model = KeyBERT(model='all-MiniLM-L6-v2')
keyword_list = []
for i in corpus.index:
    document = corpus.iloc[i]['Bewertungen']
    keywords = kw_model.extract_keywords(document)
    print(keywords)
    # nur die Worte zur Liste hinzufügen
    n = 0 
    words = [x[n] for x in keywords]
    keyword_list.append(words)
    #print(words)

#Dann das Ergebnis als .csv Speichern
keywords_df = pd.DataFrame(keyword_list)
keywords_df.to_csv('keybert-keywords.csv', sep=',', index=False)
