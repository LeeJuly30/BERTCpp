import torch
import time

from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('../model/bert-base-uncased-vocab.txt')
model = BertModel.from_pretrained('../model/')
model.eval()
model.cpu()

input_string = ["how are you! i am very happy to see you guys, please give me five ok? thanks", "this is some jokes, please tell somebody else that reputation to user privacy protection. There is no central authority or supervisor having overall manipulations over others, which makes Bitcoin favored by many. Unlike lling piles of identity information sheets before opening bank accounts, users of Bitcoin need only a pseudonym, a.k.a an address or a hashed public key, to participate the system."]
input_ids = []
masks = []
type_ids = []
for string in input_string:
    tokens = tokenizer.tokenize(string)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    mask = [1] * len(tokens)
    for _ in range(len(tokens), 128):
        tokens.append("[PAD]")
        mask.append(0)
    input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    masks.append(mask)
    type_ids.append([0]*128)

input_tensor = torch.tensor(input_ids, dtype=torch.long)
mask_tensor = torch.tensor(masks, dtype=torch.long)
type_tensor = torch.tensor(type_ids, dtype=torch.long)

start = time.time()
for _ in range(10):
    with torch.no_grad():
        output, poolout = model(input_tensor, token_type_ids=type_tensor, attention_mask=mask_tensor, output_all_encoded_layers=False)
end = time.time()
print('time:', (end-start)*1000)
print(output.size())
print(output)
print(poolout.size())
print(poolout)




    