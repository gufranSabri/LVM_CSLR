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

        if hidden_size != 768:
            self.fc = nn.Sequential(
                nn.Linear(768, hidden_size),
                nn.ReLU()
            )
    
    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.model.forward(x, self.tokenizer).float()
        else:
            x = self.model.forward(x, self.tokenizer).float()

        if hasattr(self, 'fc'):
            x = self.fc(x)

        return x
    
def test():
    texts = [
        'Three blind horses listening to Mozart.',
        'Älgen är skogens konung!',
        'Wie leben Eisbären in der Antarktis?',
        'Вы знали, что все белые медведи левши?'
    ]
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'

    model = CLIP_Text_Encoder(model_name, hidden_size=256, freeze=False)
    for text in texts:
        print(model(text).shape)

if __name__ == "__main__":
    test()