from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import csv
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
stoplist = set('i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers '
               'herself it its itself they them their theirs themselves what which who whom this that these those am is '
               'are was were be been being have has had having do does did doing a an the and but if or because as until '
               'while of at by for with about against between into through during before after above below to from up down '
               'in out on off over under again further then once here there when where why how all any both each few more '
               'most other some such no nor not only own same so than too very s t can will just don should now'.split())


def vector_for_learning(model, input_docs):
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=100)) for doc in input_docs])
    return targets, feature_vectors


# Initializing the variables and tags with numbers
stemmer = SnowballStemmer("english", ignore_stopwords=True)
wordnet_lemmatizer = WordNetLemmatizer()
train_documents = []
test_documents = []
test_movie_ids = []
kaggle_pred = []
csvData = []
a = 0
i = 1
tags_index = {'sci-fi': 1, 'action': 2, 'comedy': 3, 'fantasy': 4, 'animation': 5, 'romance': 6}

# Reading the file
FILEPATH = 'tagged_plots_movielens.csv'
with open(FILEPATH, 'r') as csvfile:
    csvReader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvReader)
    for row in csvReader:
        if row[2].lower() not in stoplist:
            if i <= 2000:
                train_documents.append(
                    TaggedDocument(words=word_tokenize((stemmer.stem(wordnet_lemmatizer.lemmatize(row[2].lower())))),
                                   tags=[tags_index.get(row[3])]))
            else:
                test_documents.append(
                    TaggedDocument(words=word_tokenize((stemmer.stem(wordnet_lemmatizer.lemmatize(row[2].lower())))),
                                   tags=[tags_index.get(row[3])]))
                test_movie_ids.append(row[1])
        i += 1

movieModel = Doc2Vec(dm=0, vector_size=200, negative=5, hs=0, min_count=2, sample=0, workers=8, alpha=0.025,
                     min_alpha=0.001)
movieModel.build_vocab([x for x in train_documents])
movieModel.train(train_documents, total_examples=len(train_documents), epochs=20)

movieModel.save('movieModel.d2v')

y_train, X_train = vector_for_learning(movieModel, train_documents)
y_test, X_test = vector_for_learning(movieModel, test_documents)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy for movie plots is %s' % accuracy_score(y_test, y_pred))

for i in y_pred:
    for k, v in tags_index.items():
        if i == v:
            kaggle_pred.append(k)

csvData.append(['ID', 'Value'])
for a in range(len(test_movie_ids)):
    csvRow = []
    csvRow.append(test_movie_ids[a])
    csvRow.append(kaggle_pred[a])
    csvData.append(csvRow)

with open('tagged_plots_movielens_test.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()
