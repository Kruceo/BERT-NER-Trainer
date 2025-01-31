from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import Trainer, TrainingArguments
from kset import load_dataset
from torch import cuda, device
from ner_dataset import NERDataset
tokenizer = BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased', use_fast=True)
model = BertForTokenClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=3)  # 3 classes: B-PER, I-PER, O

model_output="./model_output"

main_device = device("cuda" if cuda.is_available() else "cpu")  # Nome da GPU

if cuda.is_available():
    print("Using",cuda.get_device_name(0))    

tokens,labels = load_dataset('train.txt')

training_slice_p = 0.8
evaluate_slice_p = 0.1
training_slice_start = 0
training_slice_end = round(len(tokens)*training_slice_p)
evaluate_slice_start = training_slice_end
evaluate_slice_end = training_slice_end + round(len(tokens)*evaluate_slice_p)

label2id = {
    "B-PER": 0,  # Começo de uma entidade do tipo PESSOA
    "I-PER": 1,
    "O":     2       # Outros (não é uma entidade)
}

model.config.label2id = label2id

training_ds = NERDataset(tokens[training_slice_start:training_slice_end],labels[training_slice_start:training_slice_end],tokenizer,label2id)
evaluate_ds = NERDataset(tokens[evaluate_slice_start:evaluate_slice_end],labels[evaluate_slice_start:evaluate_slice_end],tokenizer,label2id)

training_args = TrainingArguments(
    output_dir="./results",  # Onde salvar os checkpoints
    eval_strategy="epoch",  # Avaliação por época
    # save_strategy="epoch",
    report_to="none",
    learning_rate=4e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=512,
    weight_decay=0.01,
    logging_dir="./logs",  # Diretório para logs
    logging_steps=50
)

# Crie o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_ds,  # Seu dataset de treino
    eval_dataset=evaluate_ds,     # Seu dataset de validação
)

# # Treine o modelo
trainer.train()

model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")