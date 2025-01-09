# %%
import torch
import string
from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART tokenizer and model
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

# Number of top predictions to consider
top_k = 10
print(f"Mask token: {bart_tokenizer.mask_token}")
print(f"Mask token ID: {bart_tokenizer.mask_token_id}")


def decode(tokenizer, pred_idx, top_clean):
    """
    Decodes predicted token indices into a string while removing unwanted tokens.
    """
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    """
    Encodes a sentence for the BART model, replacing <mask> with the tokenizer's mask token.
    """
    try:
        text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
        print(f"Text after replacing <mask>: {text_sentence}")

        # If <mask> is the last token, append a "." to prevent models from predicting punctuation.
        if tokenizer.mask_token == text_sentence.split()[-1]:
            text_sentence += ' .'

        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        print(f"Tokenized input IDs: {input_ids}")

        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
        print(f"Mask index: {mask_idx}")

        return input_ids, mask_idx
    except Exception as e:
        print(f"Error in encode function: {e}")
        raise e


def get_all_predictions(text_sentence, top_clean=5):
    """
    Gets predictions from the BART model for a sentence with a <mask>.
    """
    print(f"Input Sentence: {text_sentence}")

    try:
        # Encode the input sentence
        input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
        print(f"Encoded input IDs: {input_ids}")
        print(f"Mask index: {mask_idx}")

        # Make predictions
        with torch.no_grad():
            predict = bart_model(input_ids)[0]
        print(f"Model raw predictions: {predict.shape}")

        # Decode predictions to get the top `top_clean` results
        bart_predictions = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
        print(f"Decoded predictions: {bart_predictions}")

        return {'bart': bart_predictions}
    except Exception as e:
        print(f"Error in get_all_predictions: {e}")
        return {}

