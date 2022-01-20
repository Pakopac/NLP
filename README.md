# Learning Natural Language Processing (NLP)
https://en.wikipedia.org/wiki/Natural_language_processing \
Deep learning for text and language analysis

## Exercice 1
```
Cas pratique : implémenter un algorithme de recommandation de film en
python avec le dataset movies.csv :
https://drive.google.com/drive/folders/1GCWcIvE3ZipWiV8567CswTNUaNzce3s
T?usp=sharing
Question 1 : importer Pandas, Numpy, CountVectorizer() et cosine_similarity() et le dataset
Question 2 : Faire une fonction appelé combine_features() qui va combiner les features
['keywords','cast','genres','director'] dans un unique string
Question 3 : Utiliser la fonction fillna() de Pandas pour remplir les NaN du dataset par des
string vide et créer une nouvelle colonne appelé “combined_features” en appliquant la
fonction combine_features() sur chaque ligne du dataset.
Question 4 : Utiliser la fonction CountVectorizer() et afficher la matrice de comptage de la
nouvelle colonne combined_features.
Question 5 : Utiliser la fonction cosine_similarity() sur la matrice de comptage définie à la
question précédente et créer une variable avec.
Question 6 : Utiliser ces fonction afin de calculer le top 5 des films similaire au film Titanic
Indice 1 : utiliser la fonction get_index_from_title() sur le film demandé et utiliser ensuite la
variable définie à la question précédente.
Indice 2 : Trier la liste obtenue à l’aide du résultat de l’indice précédent.
Indice 3 : Faire une boucle sur la liste obtenue et stopper votre boucle à la 5eme itération
```
```
df = pd.read_csv(data_dir + 'movies.csv')
```
- Load csv

```
def combine_features(df):
  df["combined_features"] = df["keywords"] + df["cast"] + df["genres"] + df["director"]
```
- Concat categories we need

```
df = df.fillna("")
```
- Convert all NaN value to empty string
```
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["combined_features"]) 
```
- CountVectorizer: Convert a collection of text documents to a matrix of token counts.

```
sim = cosine_similarity(X)
```
- Compute cosine similarity in X array to check similarities between films

```
id_titanic = df.index[df['original_title'] == 'Titanic'].tolist()[0]
sim_titanic = dict(enumerate(x for x in sim[id_titanic]))
sim_titanic = {k: v for k, v in sorted(sim_titanic.items(), key=lambda item: item[1], reverse=True)}
```
- Dictionary {id: sim_value} to retrieve indexes of movies after sorting 
