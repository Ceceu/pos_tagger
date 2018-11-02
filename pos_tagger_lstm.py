import nltk
import numpy as np
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def get_resources():
    nltk.download('treebank')
    return nltk.corpus.treebank.tagged_sents()


def get_model(MAX_LENGTH, word2index, tag2index):
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH,)))
    model.add(Embedding(len(word2index), 128))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    return model


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])

        token_sequences.append(token_sequence)

    return token_sequences


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def training():
    # get resources
    nltk.download('treebank')
    tagged_sentences = nltk.corpus.treebank.tagged_sents()

    # separate the words from the tags
    sentences, sentence_tags = [], []
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))

    # separate into train, test and dev
    (train_sentences,
     test_sentences,
     train_tags,
     test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

    print(test_sentences[0])
    print(test_tags[0])

    # word and tags to number index
    words, tags = set([]), set([])

    for s in train_sentences:
        for w in s:
            words.add(w.lower())

    for ts in train_tags:
        for t in ts:
            tags.add(t)

    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs

    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # The special value used to padding

    # convert the word dataset to integer dataset, both the words and the tags
    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        train_sentences_X.append(s_int)

    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        test_sentences_X.append(s_int)

    for s in train_tags:
        train_tags_y.append([tag2index[t] for t in s])

    for s in test_tags:
        test_tags_y.append([tag2index[t] for t in s])

    #
    MAX_LENGTH = len(max(train_sentences_X, key=len))
    print("MAX_LENGTH: {}".format(MAX_LENGTH))

    #
    train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
    test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
    train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
    test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

    # keras model
    # model = get_model(MAX_LENGTH, word2index, tag2index)
    # model.summary()

    # cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
    # print(cat_train_tags_y[0])

    # train the model
    # model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=1,
    #          validation_split=0.2)

    # save model
    # model.save('data/keras_lstm.h5')

    model = load_model('data/keras_lstm.h5')

    predictions = model.predict(test_sentences_X)
    p = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
    print(p[100])

    # evaluate
    # scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
    # print(f"{model.metrics_names[1]}: {scores[1] * 100}")

    # evaluate
    # y_preds = model.predict(test_sentences_X)
    # Compute classification report
    # print(y_preds)

    # print("test tag")
    # print(test_tags_y)

    # classif_report = classification_report(y_true=test_tags_y, y_pred=test_tags_y, target_names=list(tags))
    # print(classif_report)


if __name__ == '__main__':
    training()
