import torch
import torch.nn as nn
from multilingual_clip import pt_multilingual_clip
import transformers

class CLIP_Text_Encoder(nn.Module):
    def __init__(
            self, 
            model_name,
            hidden_size=512, 
            freeze=False,
        ):
        super(CLIP_Text_Encoder, self).__init__()

        self.freeze = freeze
        self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU()
        )
    
    def custom_forward(self, x, device):
        res = []
        for text in x:
            txt_tok = self.tokenizer(text, padding=True, return_tensors='pt').to(device)
            embs = self.model.transformer(**txt_tok)[0]
            att = txt_tok['attention_mask']
            embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
            embs = self.model.LinearTransformation(embs).squeeze(0)
            res.append(embs)

        res = torch.stack(res, dim=0)
        return res

    def forward(self, x, device):
        x = self.custom_forward(x, device)
        x = self.fc(x)

        return x
    
def test():
    texts = [
        'Three blind horses listening to Mozart.',
        'Älgen är skogens konung!',
        'Wie leben Eisbären in der Antarktis?',
        'Вы знали, что все белые медведи левши?'
    ]
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'

    model = CLIP_Text_Encoder(model_name, hidden_size=256, freeze=False).to("mps")
    print(model(texts, "mps").shape)
        

if __name__ == "__main__":
    test()