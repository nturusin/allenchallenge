from gensim.models import Word2Vec

model = Word2Vec.load("300features_40minwords_10context")


print type(model.syn0)

print model.syn0.shape
