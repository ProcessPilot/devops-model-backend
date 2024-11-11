from transformers import AutoTokenizer

model_name = "microsoft/phi-2"


# Function for text generation
def gen(model, prompt, maxlen=100, sample=True):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    toks = tokenizer(prompt, return_tensors="pt")
    res = model.generate(
        **toks.to("cuda"),
        max_new_tokens=maxlen,
        do_sample=sample,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
    )
    return tokenizer.batch_decode(res, skip_special_tokens=True)
