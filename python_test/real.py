import torch
import time

from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('../model/vocab.txt')
model = BertModel.from_pretrained('../model/')
model.eval()
model.cpu()

input_string = ["同学们，今天我们来学习一个新词汇，叫做量化交易，好了我们开始吧！","因为有些算法还是不容易理解的，你得知道什么地方用什么，还得知道为啥那么用。单词就无脑背诵都记不下来，那LeetCode自然一次记不住就太正常了。其实上面的类比你懂了的话，你就知道，刷LeetCode也是无他，多刷两遍就好了，多总结总复习，常用的东西还真得背下来。"]
input_ids = []
masks = []
type_ids = []
for string in input_string:
    tokens = tokenizer.tokenize(string)
    print(tokens)
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




    