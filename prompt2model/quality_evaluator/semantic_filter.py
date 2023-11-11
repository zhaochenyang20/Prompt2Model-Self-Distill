import nltk
import torch
from transformers import AutoTokenizer, BertForNextSentencePrediction
from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("OutputAnnotator")

try:
    if not nltk.download("punkt"):
        nltk.download("punkt")
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

def batch_pairs_coherence(sentences):
    """
    Checks the coherence of consecutive sentence pairs within a list of sentences.

    Params:
        sentences (list of str): A list of sentences to check for coherence.

    Return:
        bool: True if all consecutive sentence pairs are coherent according to the model, False otherwise.
    """

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased").to(device)
    except Exception as e:
        logger.warning(f"Error loading model or tokenizer: {e}")

    try:
        sentence_pairs = [
            (sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)
        ]
        encoded_batch = tokenizer.batch_encode_plus(
            sentence_pairs, return_tensors="pt", padding=True
        )
        encoded_batch = {key: val.to(device) for key, val in encoded_batch.items()}
        with torch.no_grad():
            outputs = model(**encoded_batch)
            logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        is_next_labels = torch.argmax(probs, dim=1) == 0
        return all(is_next_labels.cpu())
    except Exception as e:
        logger.warning(f"Error in batch_pairs_coherence: {e}")
        return None


def check_paragraph_coherence(paragraphs):
    """
    Splits a paragraph into sentences and checks the overall coherence by evaluating each consecutive sentence pair.

    Params:
        paragraphs (list of str): A list containing the paragraphs to check for coherence.

    Return:
        filtered_paragraphs (list of str): A list containing the coherent paragraphs.
    """

    try:
        filtered_paragraphs = []
        for paragraph in paragraphs:
            sentences = split_long_sentences(paragraph)
            if len(sentences) < 2:
                filtered_paragraphs.append(paragraph)
            elif batch_pairs_coherence(sentences):
                filtered_paragraphs.append(paragraph)
        return filtered_paragraphs
    except Exception as e:
        logger.warning(f"Error in check_paragraph_coherence: {e}")
        return None


def split_long_sentences(paragraph, max_words=50):
    """
    Splits a paragraph into sentences and then further splits any sentence exceeding a specified maximum word count.
    This ensures that each sentence or sentence fragment in the output is below the maximum word count.

    Params:
        paragraph (str): The paragraph to be split into sentences.
        max_words (int, optional): The maximum number of words allowed in each sentence or sentence fragment. Defaults to 50.

    Return:
        split_sentences (list of str): A list containing sentences and sentence fragments, each not exceeding the maximum word count.
    """
    sentences = nltk.tokenize.sent_tokenize(paragraph)
    split_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            split_sentences.append(sentence)
        else:
            for i in range(0, len(words), max_words):
                part = ' '.join(words[i:i+max_words])
                split_sentences.append(part)
    return split_sentences
