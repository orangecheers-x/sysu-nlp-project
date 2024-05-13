from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "<len> Who are you? <text> ok"
encoded_input = tokenizer.encode(text, return_tensors='pt')
tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<len>', '<text>']})

print(tokenizer.tokenize(text))
print(tokenizer.convert_tokens_to_ids(
    tokenizer.tokenize(text, add_special_tokens=True)))
print(tokenizer.special_tokens_map)
model.resize_token_embeddings(len(tokenizer))
output = model.generate(encoded_input)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
