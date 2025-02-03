from torch.utils.data import Dataset
from torch import tensor

class NERDataset(Dataset):
    def __init__(self, tokens, labels, tokenizer, label2id):
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        labels = self.labels[idx]
        
        # Tokenizar com truncamento e padding
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        aligned_labels = self.align_labels_with_tokens(labels, encoding)
        
        # Retornar em formato tensor
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": tensor(aligned_labels)
            }

    def to_decoded(self):
        
        parsed = []
        for sentence in self.tokens:
            sentence_string = ""
            for tkn_index in range(len(sentence)):
                if tkn_index == len(sentence)-1:
                    sentence_string += sentence[tkn_index]
                else:
                    sentence_string += sentence[tkn_index] + " "
            parsed.append(sentence_string)
        return parsed

    
    def align_labels_with_tokens(self, labels, encoding):
        # quando é feita a tokenização de uma frase
        # o tokenizer pode gerar tokens a mais da frase original
        # como marcadores de inicio e fim, subtokens divindo palavras em 2 ou mais tokens, entre outros...
        # essa funcao alinha os labels com esse novos tokens 
        # sem isso provavelmente o tamanho dos labels e dos tokens seriam diferentes, e daria erro
        # Obter IDs de tokens após tokenização
        word_ids = encoding.word_ids()
        
        # Criar lista para rótulos alinhados
        aligned_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:  # Padding
                aligned_labels.append(-100)  # Ignorar na perda
            elif word_id != current_word:  # Novo token (começo de uma nova palavra)
                aligned_labels.append(self.label2id.get(labels[word_id], -100))
                current_word = word_id
            else:  # Subtoken
                aligned_labels.append(self.label2id.get(labels[word_id], -100))
        return aligned_labels

