import openai
from keybert.llm import OpenAI
from keybert import KeyLLM
import csv
import pandas as pd


#Corpus aus dem csv laden, wobei jede Zeile ein Dokument ist
file = open("fokussierte_Bewertungen.csv", "r")
corpus = [line.strip() for line in file]
file.close()
num_documents = len(corpus)

#Kontrolle, ob Zeilen des .csv mit Zeilen im Dataframe übereinstimmen
print("Number of Rows (documents) in corpus: ", str(num_documents))

#Festlegen wieviele Queries auf Einmal an die OpenAI API gesendet werden
#Max Tokens for ChatGPT 3.5 Turbo is 4096
num_api_queries = 200
denominator, rest = divmod(num_documents, num_api_queries)
print("Number of documents every query: ", str(num_api_queries))
print("Number of queries to OpenAI API: ", str(denominator))
print("Remaining documents: ", str(rest))

#OpenAi Bibliothek laden und mit dem eigenen API Key initialisieren
client = openai.OpenAI(api_key="insert API Key here")
llm = OpenAI(client)
kw_model = KeyLLM(llm)

# Extract keywords mit Hilfe des LLM von OpenAi
# Dabei werden vom LLM Keywords zurückgegeben welche das Dokument beschreiben
# Zuerst die Schleife so lange durchlaufen bis Zähler mal Anzahl Dokumente pro query abgearbeitet sind
corpus_part = []
keywords_llm = []
x = 1
for y in range(1, denominator):
    z = y * num_api_queries    
    corpus_part = kw_model.extract_keywords(corpus[x:z])
    keywords_llm.append(corpus_part)
    x = z + 1

#Dann die restlichen Dokumente    
remaining_docs_start = num_api_queries * denominator + 1
index_last_doc = num_documents - 1
corpus_part = kw_model.extract_keywords(corpus[remaining_docs_start:index_last_doc])
keywords_llm.append(corpus_part)

#Dann das Ergebnis als .csv Speichern
keywords_df = pd.DataFrame(keywords_llm)
keywords_df.to_csv('openai-llm-keywords.csv', sep=',', index=False)
