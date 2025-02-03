from transformers import pipeline, BertForTokenClassification, BertTokenizerFast
import utils
from torch.utils.data import DataLoader
from datasets import Dataset
from kset import load_dataset
from ner_dataset import NERDataset
# Carregar modelo e tokenizer locais
model_name = "./model_output"
model = BertForTokenClassification.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name,use_fast=True)


# Carregar a pipeline para NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Teste com um exemplo de texto
tokens,labels = load_dataset("train.txt")
tokens = tokens[round(0.9*len(tokens)):]
labels = labels[round(0.9*len(labels)):]

label2id = {
    "B-PER": 0,  
    "I-PER": 1,
    "O":     2 
}

sentences = NERDataset(tokens,labels, tokenizer,label2id)

decoded_sentences = sentences.to_decoded()

dataloader = DataLoader(decoded_sentences, batch_size=32)

results = []
for batch in dataloader:
    # print(batch)
    results.extend(nlp(batch))

total_tokens = 0
total_corrects = 0
for result_index in range(len(results)):
    # print(result)
    sentence_correct_labels = sentences[result_index]['labels'][1:]
    
    total = len(results[result_index])
    total_tokens += total
    correct = 0
    
    for token_res_index in range(len(results[result_index])):
        result_label = results[result_index][token_res_index]['entity']
        correct_label = f"LABEL_{sentence_correct_labels[token_res_index]}"
        
        if result_label == correct_label:
            # print(utils.bcolors.OKGREEN,result_label,"vs",correct_label,results[result_index][token_res_index]['word'],utils.bcolors.ENDC)
            correct +=1 
        # else:
            # print(utils.bcolors.FAIL,result_label,"vs",correct_label,results[result_index][token_res_index]['word'],utils.bcolors.ENDC)
            
    # print(tokenizer.tokenize(decoded_sentences[result_index]))
    print("")
    utils.pretty_print_results(results[result_index])
    total_corrects += correct
    print(f"{correct}/{total}","correct classifications")
    print("\n=================================")

print(f"{total_corrects}/{total_tokens} correct classifications")