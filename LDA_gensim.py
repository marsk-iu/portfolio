import spacy, pandas as pd, cld3
import gensim, pyLDAvis, pyLDAvis.gensim_models
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint

#Deutsches Sprachmodel für Spacy
nlp = spacy.load("de_core_news_sm")

#Corpus aus dem csv laden, wobei jede Zeile ein Dokument ist
corpus = pd.read_csv('fokussierte_Bewertungen.csv')

#Kontrolle, ob Zeilen des .csv mit Zeilen im Dataframe übereinstimmen
print('Number of Rows in DataFrame: ', len(corpus.index))

#Funktion um Token zu filtern
def is_token_allowed(token):
     return bool(
         token
         and str(token).strip()
         and not token.is_stop
         and not token.is_punct
         and not len(token) < 2
# weitere Möglichkeit zum Experimentieren, nämlich nur Token zulassen, die nicht nur aus Zahlen bestehen
#         and token.is_alpha
     )

#Funktion zum Lemmatisieren und in Kleinbuchstaben umwandeln    
def preprocess_token(token):
     return token.lemma_.strip().lower()
   
#Jedes Dokument (Zeile im corpus) in token umwandeln und daraus ein Liste erstellen welche jeweils eine Liste von Tokens des entsprechenden Dokuments enthält    
document_list = []
for i in corpus.index:
    document = nlp(corpus.iloc[i]['Bewertungen'])
    filtered_tokens = [
     preprocess_token(token)
     for token in document
    if is_token_allowed(token)
    ]
    document_list.append(filtered_tokens)

#print("Number of lines in document_list: " + str(len(document_list)))

# Bigramme erstellen mit mindestvorkommen 5
bigram = gensim.models.Phrases(document_list, min_count=5)
#print(bigram)

for idx in range(len(document_list)):
    for token in bigram[document_list[idx]]:
        if '_' in token:
            # Wenn ein Token einen _ enthält handelt es sich um ein Bigram, daher an liste anhängen
            document_list[idx].append(token)
                       
# Wörterbuch erstellen
id2word = corpora.Dictionary(document_list)

# Wörter filtern die weniger als 20 Mal vorkommen und in mehr als 50% der Dokumente vorkommen
id2word.filter_extremes(no_below=20, no_above=0.5)

# Jede Zeile in der Liste ist ein Dokument
documents = document_list

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
    pyLDAvis.save_html(vis, str(num_top)+'_lda_result.html')

# Coherence Score berechnen
    coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nNuber of Topics: ', num_top)
    print('Coherence Score: ', coherence_lda)
    
# Modell Ergebnis ausgeben
    pprint(lda_model.print_topics())

    