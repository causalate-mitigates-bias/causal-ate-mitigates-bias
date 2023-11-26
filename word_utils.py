import time, re, numpy as np, csv
import spacy, nltk, string, emoji

from transformers import RobertaTokenizer, RobertaModel, pipeline

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer

# We will use the spacy tokenizer
# https://huggingface.co/docs/transformers/tokenizer_summary
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = English()
tokenizer = Tokenizer(nlp.vocab)

# The nltk tokenizer automatically splits into an array. We will use this apart from spacy
from nltk.tokenize import word_tokenize


# We can try Stop words from nltk
def getNLTKStopWords():
    # nltk.download('stopwords')
    # nltk.download('punkt')
    nltkStopwords = nltk.corpus.stopwords.words('english')
    # print(nltkStopwords)
    return nltkStopwords


# Or use stop words from spacy
def getSpacyStopWords():
    # first run the following command to download english library
    # python -m spacy download en
    sp = spacy.load('en_core_web_sm')
    # Get the stop words
    all_stopwords = sp.Defaults.stop_words
    try:
        all_stopwords.remove('not')
    except:
        pass
    return all_stopwords


# We normalize the input text into some standard form, with lowercase, no accents and whitespace removed
# https://huggingface.co/docs/tokenizers/api/normalizers
def normalizeSentence(inputText, normalizerSequence):
    normalizer = normalizers.Sequence(normalizerSequence)
    normalizedInput = normalizer.normalize_str(inputText)
    return normalizedInput


def convertStringToList(inputString):
    outList = []
    outList[:0] = inputString
    return outList


def getStopWords():
    all_stopwords = getNLTKStopWords()
    all_stopwords.extend(convertStringToList(string.punctuation))
    all_stopwords.extend(list(emoji.UNICODE_EMOJI['en'].keys()))
    return all_stopwords


def getMaskedSentences(sentence, normalizerSequence, unmaskerModel, originalSentenceID=0, labeledData=False, label=0,
                       outputSentencesStartCounter=0):
    normalizedSentence = normalizeSentence(sentence, normalizerSequence)
    normalizedSentenceTokens = word_tokenize(normalizedSentence)
    all_stopwords = getStopWords()
    # print(all_stopwords)
    outputList = []
    sentencesCounter = 0
    for wordCounter in range(len(normalizedSentenceTokens)):
        normalizedSentenceToken = normalizedSentenceTokens[wordCounter]
        if normalizedSentenceToken not in all_stopwords:
            predictedSentences = []
            tokensWithMask = maskWordinTokensList(normalizedSentenceTokens, wordCounter)
            maskedSentence = " ".join(tokensWithMask)
            results = unmaskerModel(maskedSentence)
            for resultCounter in range(len(results)):
                newsentence = results[resultCounter]['sequence']
                newsentence = normalizeSentence(newsentence, normalizerSequence)
                if newsentence != normalizedSentence and newsentence not in predictedSentences:
                    # print("newsentence=", newsentence)
                    if labeledData:
                        # Saved as 0. Sentence ID 1. Original Sentence ID 2. new sentence 3. word masked 4. word Location 5. label
                        addTuple = (sentencesCounter + outputSentencesStartCounter, originalSentenceID, newsentence,
                                    normalizedSentenceToken, wordCounter, label)
                    else:
                        # Saved as 0. Sentence ID 1. Original Sentence ID 2. new sentence 3. word masked 4. word Location
                        addTuple = (sentencesCounter + outputSentencesStartCounter, originalSentenceID, newsentence,
                                    normalizedSentenceToken, wordCounter)
                    outputList.append(addTuple)
                    sentencesCounter += 1
                    predictedSentences.append(newsentence)
                else:
                    # print("newsentence same as old=", newsentence)
                    pass
    return outputList


def read_text(filePath, filterWords=None, labeled_data=False, delimiter="|"):
    if filterWords is None:
        filterWords = []

    with open(filePath, encoding="utf8") as input_file:
        reader = csv.reader(input_file, delimiter=delimiter)
        # reader = csv.reader(input_file, delimiter=delimiter)
        next(reader, None)  # skip the first line
        outData = []
        textColumn = 0
        for line in reader:
            cleanedText = line[textColumn]
            cleanedText = ' '.join([word for word in cleanedText.split() if word not in filterWords])
            cleanedText = re.sub(r'#([^ ]*)', r'\1', cleanedText)
            cleanedText = re.sub(r'https.*[^ ]', 'URL', cleanedText)
            cleanedText = re.sub(r'http.*[^ ]', 'URL', cleanedText)
            cleanedText = re.sub(r'@([^ ]*)', '@USER', cleanedText)
            cleanedText = emoji.demojize(cleanedText)
            cleanedText = re.sub(r'(:.*?:)', r' \1 ', cleanedText)
            cleanedText = re.sub(' +', ' ', cleanedText)
            outData.append(cleanedText)

    return outData


def read_data_csv(filePath, filterWords=None, labeled_data=False, delimiter="|", sentenceIdColumn=0, textColumn=1,
                  labelsColumn=2):
    if filterWords is None:
        filterWords = []

    with open(filePath, encoding="utf8") as input_file:
        reader = csv.reader(input_file, delimiter=delimiter)
        # reader = csv.reader(input_file, delimiter=delimiter)
        next(reader, None)  # skip the first line
        outData = []
        for line in reader:
            cleanedText = line[textColumn]
            cleanedText = ' '.join([word for word in cleanedText.split() if word not in filterWords])
            cleanedText = re.sub(r'#([^ ]*)', r'\1', cleanedText)
            cleanedText = re.sub(r'https.*[^ ]', 'URL', cleanedText)
            cleanedText = re.sub(r'http.*[^ ]', 'URL', cleanedText)
            cleanedText = re.sub(r'@([^ ]*)', '@USER', cleanedText)
            cleanedText = emoji.demojize(cleanedText)
            cleanedText = re.sub(r'(:.*?:)', r' \1 ', cleanedText)
            cleanedText = re.sub(' +', ' ', cleanedText)
            if labeled_data:
                outData.append((line[sentenceIdColumn], cleanedText))
            else:
                outData.append((line[sentenceIdColumn], cleanedText, line[labelsColumn]))
    return outData


if __name__ == '__main__':
    startTime = time.time()
    original = " Héllò hôw are ü? 😃 is a smiley "

    # https://huggingface.co/docs/tokenizers/api/normalizers
    normalizerSequence = [NFD(), StripAccents(), Strip(), Lowercase()]

    normalizedSentence = normalizeSentence(original, normalizerSequence)
    print("normalizedSentence=", normalizedSentence)

    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)
