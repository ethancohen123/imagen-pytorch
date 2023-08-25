import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorWithPadding
from typing import List
from transformers import RobertaConfig



# config
DEFAULT_ROBERTA_NAME = 'entropy/roberta_zinc_480m'
MAX_LENGTH = 128

ROBERTA_CONFIGS = {}

# singleton globals
def get_tokenizer(name):
    tokenizer = RobertaTokenizerFast.from_pretrained(name, max_len=MAX_LENGTH)
    return tokenizer

def get_model(name):
    model = RobertaForMaskedLM.from_pretrained(name)
    return model

def get_model_and_tokenizer(name):
    global ROBERTA_CONFIGS

    if name not in ROBERTA_CONFIGS:
        ROBERTA_CONFIGS[name] = dict()
    if "model" not in ROBERTA_CONFIGS[name]:
        ROBERTA_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in ROBERTA_CONFIGS[name]:
        ROBERTA_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return ROBERTA_CONFIGS[name]['model'], ROBERTA_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
    if name not in ROBERTA_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = RobertaConfig.from_pretrained(name)
        ROBERTA_CONFIGS[name] = dict(config=config)
    elif "config" in ROBERTA_CONFIGS[name]:
        config = ROBERTA_CONFIGS[name]["config"]
    elif "model" in ROBERTA_CONFIGS[name]:
        config = ROBERTA_CONFIGS[name]["model"].config
    else:
        assert False
    return config.hidden_size

def roberta_encode_text(
    texts: List[str],
    name=DEFAULT_ROBERTA_NAME,
    return_attn_mask=False
):
    model, tokenizer = get_model_and_tokenizer(name)
    collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

    inputs = collator(tokenizer(texts))
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        full_embeddings = outputs[1][-1]

    mask = inputs['attention_mask']
    embeddings = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1))

    if return_attn_mask:
        return embeddings, mask

    return embeddings
