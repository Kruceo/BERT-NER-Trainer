from transformers import EarlyStoppingCallback,get_scheduler, AdamW,BertTokenizerFast, BertForTokenClassification,Trainer, TrainingArguments
from kset import load_dataset
from torch import cuda, device
from ner_dataset import NERDataset

model_name = 'neuralmind/bert-base-portuguese-cased'
# model_name = './model_output'
tokenizer = BertTokenizerFast.from_pretrained(model_name, use_fast=True)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=3)  # 3 classes: B-PER, I-PER, O

model_output="./model_output"

main_device = device("cuda" if cuda.is_available() else "cpu")  # Nome da GPU


print("Using",cuda.get_device_name(0))    

tokens,labels = load_dataset('train.txt')

training_slice_p = 0.70
evaluate_slice_p = 0.20
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
    eval_strategy="steps",  # Avaliação por época
    
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss", 
    greater_is_better=False,
    
    report_to="none",
    learning_rate=8e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=256,
    weight_decay=0.5,
    logging_dir="./logs",  # Diretório para logs
    logging_steps=10
)

optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.08)

lr_scheduler = get_scheduler(
    "cosine",  
    optimizer=optimizer,
    num_warmup_steps=256, 
    num_training_steps=len(training_ds) 
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_ds, 
    eval_dataset=evaluate_ds,
    optimizers=(optimizer,lr_scheduler)
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

# # Treine o modelo
trainer.train()

model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")