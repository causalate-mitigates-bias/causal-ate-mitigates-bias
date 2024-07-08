import numpy as np
import time
from utilities.clean_data import clean_data
import nltk, string
from nltk.tokenize import word_tokenize
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase
from keras_preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset

from word_utils import normalizeSentence
from tqdm import tqdm

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    BertForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer
)

MODEL_CLASSES = {
    # "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta-base": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    # "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    # "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "bert-classifier": (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def expand_df_with_masked_words(inputDF):
    inputDFWithMaskedSentences = inputDF.explode(['masked_words', 'masked_sentences'])
    inputDFWithMaskedSentences['input_id'] = inputDFWithMaskedSentences.index.values
    inputDFWithMaskedSentences['masked_sentence_id'] = range(len(inputDFWithMaskedSentences))
    inputDFWithMaskedSentencesCols = inputDFWithMaskedSentences.columns.tolist()
    inputDFWithMaskedSentencesCols = inputDFWithMaskedSentencesCols[-1:] + inputDFWithMaskedSentencesCols[
                                                                           -2:-1] + inputDFWithMaskedSentencesCols[:-2]
    inputDFWithMaskedSentences = inputDFWithMaskedSentences[inputDFWithMaskedSentencesCols]
    inputDFWithMaskedSentences = inputDFWithMaskedSentences.set_index('masked_sentence_id')
    return inputDFWithMaskedSentences


# We can try Stop words from nltk
def get_NLTK_stop_words():
    # If not downloaded, download the 'punkt' and 'stopwords' dictionaries.
    # Check if 'stopwords' is already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # If not found, download stopwords
        nltk.download('stopwords')

    # Check if 'punkt' is already downloaded
    # try:
    #     nltk.data.find('corpora/punkt')
    # except LookupError:
    #     # If not found, download punkt
    #     nltk.download('punkt')

    nltkStopwords = nltk.corpus.stopwords.words('english')
    # print(nltkStopwords)
    return nltkStopwords


def convert_string_to_list(inputString):
    outList = []
    outList[:0] = inputString
    return outList



def getStopWords():
    all_stopwords = get_NLTK_stop_words()
    all_stopwords.extend(convert_string_to_list(string.punctuation))
    # all_stopwords.extend(list(emoji.UNICODE_EMOJI['en'].keys()))
    return all_stopwords


class SentenceDataBatcher:
    def __init__(self, batch_size, max_seq_len=100):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len  # todo: will be used later

    def prepare_batches(self, encoded_inputs):
        encoded_attention_masks = []
        for sent in encoded_inputs:
            att_mask = [int(token_id > 0) for token_id in sent]
            encoded_attention_masks.append(att_mask)  # Store the attention mask for this sentence.
        prediction_inputs = torch.tensor(encoded_inputs)
        prediction_masks = torch.tensor(encoded_attention_masks)  ## Attention masks
        prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        pred_dataloader = DataLoader(prediction_data, sampler=prediction_sampler,
                                     batch_size=self.batch_size)  # Create the DataLoader.
        return pred_dataloader


def initialize_tokenizer(input_model, tokenizerClass):
    tokenizer = tokenizerClass.from_pretrained(input_model)
    tokenizer.padding_side = "left"
    # tokenizer.eos_token = 1
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    return tokenizer


def get_masked_sentences_as_list(inputSentence, normalizer_sequence=None):
    if normalizer_sequence is None:
        normalizer_sequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    normalizedSentence = normalizeSentence(inputSentence, normalizer_sequence)
    all_stopwords = getStopWords()
    tokenizedInput = word_tokenize(normalizedSentence)
    maskedSentences = []
    for wordCounter in range(len(tokenizedInput)):
        newSentenceTokens = tokenizedInput.copy()
        normalizedSentenceToken = tokenizedInput[wordCounter]
        if normalizedSentenceToken not in all_stopwords:
            oldWord = newSentenceTokens[wordCounter]
            newSentenceTokens[wordCounter] = "<mask>"
            maskedSentence = " ".join(newSentenceTokens)
            maskedSentences.append((oldWord, maskedSentence))
    return maskedSentences


def normalize_sentences_mask_words(inputDF, normalizer_sequence=None):
    if normalizer_sequence is None:
        normalizer_sequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    inputDF['masked_outputs'] = [get_masked_sentences_as_list(sentence, normalizer_sequence) for sentence in
                                 inputDF['text']]
    inputDF['masked_words'] = [[masked_output[0] for masked_output in row] for row in inputDF['masked_outputs']]
    inputDF['masked_sentences'] = [[masked_output[1] for masked_output in row] for row in inputDF['masked_outputs']]
    inputDF.drop('masked_outputs', axis=1, inplace=True)
    return inputDF


def initialize_model(input_model_path='roberta-base', input_config_path='roberta-base', config_class=RobertaConfig,
                    model_class=RobertaForMaskedLM, cuda=True, local_files_only=False):
    config = config_class.from_pretrained(input_config_path, cache_dir=None, local_files_only=local_files_only)
    model = model_class.from_pretrained(
        input_model_path,
        from_tf=False,
        config=config,
        local_files_only=local_files_only
    )
    # model = GPT2LMHeadModel.from_pretrained(input_modelOrPath, return_dict=True)
    # Running on cpu for checks
    # cuda=False #ok working. So will comment out.
    if cuda:
        model.cuda()
    return model


# Based on the pipeline from huggingface: https://huggingface.co/transformers/v4.11.3/_modules/transformers/pipelines/fill_mask.html
def get_preds_for_masked_sentences(maskedLinesList, input_model='roberta-base', batch_size=32, max_length=100, top_k=5):
    config_class, model_class, tokenizerClass = MODEL_CLASSES[input_model]
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    input_model = input_model
    tokenizer = initialize_tokenizer(input_model, tokenizerClass)
    model = initialize_model(input_model_path=input_model, input_config_path=input_model, config_class=config_class,
                            model_class=model_class, cuda=True, local_files_only=False)

    batch_size = batch_size

    encoded_sentences = [tokenizer.encode(sentence, add_special_tokens=True, max_length=100, truncation=True) for
                         sentence in maskedLinesList]
    encoded_inputs_padded = pad_sequences(encoded_sentences, maxlen=100, dtype="long", value=tokenizer.pad_token_id,
                                          truncating="pre", padding="pre")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_batcher = SentenceDataBatcher(batch_size=batch_size)
    masked_sentence_dataloader = data_batcher.prepare_batches(encoded_inputs_padded)
    unmasked_sentence_outputs = []
    top_k = top_k
    predicted_sentences = []
    batch_num = 0
    for batch in tqdm(masked_sentence_dataloader):
        batch_num += 1
        batch = tuple(batchElem.to(device) for batchElem in batch)
        input_ids, input_mask = batch
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        output_logits = outputs["logits"]
        # print("input_ids=",input_ids)

        # print("len(input_ids)=",len(input_ids))
        # print("input_ids.shape=",input_ids.shape)
        # print("output_logits.shape=",output_logits.shape)
        masked_indices_for_batch = torch.zeros(len(input_ids), dtype=int)
        mi1, mi2 = torch.where(torch.Tensor(input_ids == tokenizer.mask_token_id))
        for i in range(len(mi2)):
            masked_indices_for_batch[mi1[i]] = mi2[i]
        try:
            assert (len(masked_indices_for_batch) == len(input_ids))
        except:
            print(
                "THE BATCH %d HAS FAILED AS MASK WAS NOT CORRECTLY FOUND. WE HAVE NOT FAILED AND INSTEAD CONTINUED PREDICTIONS, BUT VALUES ARE INCORRECT" % (
                    batch_num))
            masked_indices_for_batch = torch.tensor([0] * len(input_ids))

        out_sentences = []
        # print("torch.where(torch.Tensor(input_ids == tokenizer.mask_token_id))",torch.where(torch.Tensor(input_ids == tokenizer.mask_token_id)))
        # print("len(masked_indices_for_batch)=",len(masked_indices_for_batch))
        for i in range(len(input_ids)):
            # print("In here")
            logits_for_sent_in_batch = output_logits[i, masked_indices_for_batch[i], :]
            probs = logits_for_sent_in_batch.softmax(dim=0)
            values, predictions = probs.topk(top_k)
            encoded_sentence = input_ids[i, :]
            out_for_masked_sentence = []
            for pred in predictions.tolist():
                enc_sentence_copy = encoded_sentence.clone()
                enc_sentence_copy[masked_indices_for_batch[i]] = pred
                enc_sentence_copy = enc_sentence_copy[torch.where(encoded_sentence != tokenizer.pad_token_id)]
                pred_sentence = tokenizer.decode(enc_sentence_copy, skip_special_tokens=True)
                if not pred_sentence:
                    pred_sentence = ""
                out_for_masked_sentence.append(pred_sentence)
            # print("out_for_masked_sentence",out_for_masked_sentence)
            if len(out_for_masked_sentence) == top_k:
                out_sentences.append(out_for_masked_sentence)
            else:
                out_sentences.append(["empty_sentence"] * 5)
        predicted_sentences.extend(out_sentences)
        # print("len(input_ids)=",len(input_ids))
        # print("len(out_sentences)=",len(out_sentences))
        try:
            assert (len(out_sentences) == len(input_ids))
        except:
            raise AssertionError("THE BATCH %d HAS FAILED." % batch_num)
    return predicted_sentences


def create_masked_replacements(input_df, outfile_masked="outputs/gao_masked.csv",
                               outfile_unmasked="outputs/gao_unmasked.csv", normalizer_sequence=None,
                               replacement_model='roberta-base'):
    if normalizer_sequence is None:
        normalizer_sequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    output_df = normalize_sentences_mask_words(input_df, normalizer_sequence)
    output_df_with_masked_sentences = expand_df_with_masked_words(output_df)
    output_df_with_masked_sentences = output_df_with_masked_sentences.fillna("")
    # print(gao_dataWithMaskedSentences.head(5))
    output_df_with_masked_sentences.to_csv(outfile_masked, sep='|')

    # https://huggingface.co/roberta-base
    top_k = 5  # https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.FillMaskPipeline
    input_model = replacement_model
    # input_model = 'bert'
    batch_size = 32
    batch_size = 128
    max_length = 512
    masked_sentences_list = output_df_with_masked_sentences['masked_sentences'].to_list()
    # BELOW IS A LINE WITH NO MASKS. USED FOR DEBUGGING
    # maskedLinesList = [inputDFWithMaskedSentences['masked_sentences'].to_list()[3334]]
    unmasked_list_of_sentences = get_preds_for_masked_sentences(masked_sentences_list, input_model=input_model,
                                                         batch_size=batch_size, max_length=max_length, top_k=top_k)
    output_df_with_masked_sentences["unmasked_sentences"] = unmasked_list_of_sentences
    output_df_with_unmasked_sentences = output_df_with_masked_sentences.explode('unmasked_sentences')
    output_df_with_unmasked_sentences.to_csv(outfile_unmasked, sep='|')
    return output_df_with_masked_sentences, output_df_with_unmasked_sentences


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducability
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    normalizer_sequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    gao_data = clean_data(filename='gao')
    gao_data_masked, gao_data_unmasked = create_masked_replacements(input_df=gao_data,
                                                                    outfile_masked="outputs/gao_masked.csv",
                                                                    outfile_unmasked="outputs/gao_unmasked.csv",
                                                                    normalizer_sequence=normalizer_sequence,
                                                                    replacement_model='roberta-base')



    zampieri_data = clean_data(filename='zampieri')

    zampieri_data_masked, zampieri_data_unmasked = create_masked_replacements(input_df=zampieri_data,
                                                                    outfile_masked="outputs/zampieri_masked.csv",
                                                                    outfile_unmasked="outputs/zampieri_unmasked.csv",
                                                                    normalizer_sequence=normalizer_sequence,
                                                                    replacement_model='roberta-base')

    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))

