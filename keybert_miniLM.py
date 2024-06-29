import csv
import ast
import gensim, pyLDAvis, pyLDAvis.gensim_models
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
import pandas as pd

#Corpus aus dem csv laden, wobei jede Zeile die Stichworte zu einem Dokument enthält
keywords_llm = []

# CSV file laden welches durch ausführen von keybert_keywords.py erzeugt wurde
with open('keybert-keywords.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    
    # Zeile mit Spaltennamen überspringen
    next(csv_reader)
    
    # Schleife um alle Einträge zu durchlaufen
    for row in csv_reader:
        keywords_llm.append(row)
        #print(row)

#Wörterbuch mit allen Keywords erstellen
id2word = corpora.Dictionary()
for i in keywords_llm:
    id2word.add_documents([i])
    
# Wörter filtern die weniger als 20 Mal vorkommen und in mehr als 50% der Dokumente vorkommen
# Habe ich am Ende nicht verwendet da im fokussierten Datenset zu wenige vorhanden waren um das
# Ergebnis zu verbessern

#id2word.filter_extremes(no_below=10, no_above=0.8)

# Jede Zeile in der Liste ist ein Dokument
documents = keywords_llm

# BOW erstellen
corpus_preprocessed = [id2word.doc2bow(document) for document in documents]

print('Number of unique tokens: %d' % len(id2word))
print('Number of documents: %d' % len(documents))

#LDA Model erstellen
#Anzahl der Topics für die jeweils das Modell erstellt werden soll
number_topics = [3, 5, 7, 10, 15]

#Modell erstellen mit jeweils 20 Durchläufen
for num_top in number_topics:
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_preprocessed,id2word=id2word,num_topics=num_top,passes=20,alpha='auto',per_word_topics=True,eval_every=None)

# Visuelle Darstellung als .html file mit pyLDAvis für das Modell erstellen und Filename beginnend mit Anzahl der Topics speichern
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus_preprocessed, id2word)
    pyLDAvis.save_html(vis, str(num_top)+'_keybert_miniLM.html')

# Coherence Score berechnen
    coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nNuber of Topics: ', num_top)
    print('Coherence Score: ', coherence_lda)
    
# Modell Ergebnis ausgeben
    pprint(lda_model.print_topics())
