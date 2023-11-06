from transformers import AutoTokenizer, BertForNextSentencePrediction
import torch
import nltk

if not nltk.download('punkt'):
    nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased').to(device)

def batch_pairs_coherence(sentences):
    """
    Checks the coherence of consecutive sentence pairs within a list of sentences.
    
    Params:
        sentences (list of str): A list of sentences to check for coherence.
        
    Return:
        bool: True if all consecutive sentence pairs are coherent according to the model, False otherwise.
    """
    sentence_pairs = [(sentences[i], sentences[i+1]) for i in range(len(sentences)-1)]
    encoded_batch = tokenizer.batch_encode_plus(sentence_pairs, return_tensors='pt', padding=True)
    encoded_batch = {key: val.to(device) for key, val in encoded_batch.items()}  
    with torch.no_grad():
        outputs = model(**encoded_batch)
        logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    is_next_labels = torch.argmax(probs, dim=1) == 0
    return all(is_next_labels.cpu())

def check_paragraph_coherence(paragraphs):
    """
    Splits a paragraph into sentences and checks the overall coherence by evaluating each consecutive sentence pair.
    
    Params:
        paragraphs (list of str): A list containing the paragraphs to check for coherence.
        
    Return:
        filtered_paragraphs (list of str): A list containing the coherent paragraphs.
    """
    filtered_paragraphs = []
    for paragraph in paragraphs:
        sentences = nltk.tokenize.sent_tokenize(paragraph)
        if len(sentences) < 2:
            filtered_paragraphs.append(paragraph)
        elif batch_pairs_coherence(sentences):
            filtered_paragraphs.append(paragraph)
    return filtered_paragraphs
