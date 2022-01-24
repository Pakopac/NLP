# Learning Natural Language Processing (NLP)
![illu_intro_NLP_blog-18-1](https://user-images.githubusercontent.com/33722914/150346520-1f10f67c-da13-40b4-8634-e76caa77e075.png)
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

## Exercice 2 - FastText
```
1.Se renseigner sur la librairie fasttext
2.Aller sur colab et installer la librairie
3.Charger et nettoyer rapidement le dataset labeled_data.csv
(https://drive.google.com/drive/folders/1GCWcIvE3ZipWiV8567CswTNUaNzce3sT?u
sp=sharing )
4.Se renseigner sur la manière de labelliser les data avec fasttext
5.Entrainer un modele supervisé de fastext avec les data ci-dessus
6.Afficher les mots que le modèle a appris
7.Donner la représentation vectorielle du mot 'guy' à l’aide du modèle que vous avez
entraîné ci-dessus.
```
- FastText is a NLP library for train model in text representation and text classification 
```
!git clone https://github.com/facebookresearch/fastText.git /content/drive/MyDrive/Colab_Notebooks/fastText
!cd /content/drive/MyDrive/Colab_Notebooks/fastText; sudo pip install .
```
- Install fastText in colab

```
df = pd.read_csv(data_dir + 'labels.csv', usecols=['class', 'tweet'])
df['tweet'] = df['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))
```
- Clean dataset

```
df['class'] = '__label__' + df['class'].astype('str')
```
- Set \_\_label__ in front of our labels to labelize it

```
df.to_csv(r'/content/drive/My Drive/Colab_Notebooks/tweets_updated.txt', sep=' ', quoting=csv.QUOTE_NONE, escapechar=" ")
```
- Register our dataset with labels and text in txt file with to_csv function
- sep: space character for separator
- quoting = csv.QUOTE_NONE: deletes quotes
- escapechar: delete space char

```
model = ft.train_supervised(input='/content/drive/My Drive/Colab_Notebooks/tweets_updated.txt')
```
- Train model

```
model.predict("hello my name is lilian")
```
- Make a prediction

```
model.words
```
- Get words learned by model

```
model.get_word_vector("guy")
```
- Get vector representation