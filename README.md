# BERT-base-pt-BR-person

Model is available in [rafola/BERT-base-pt-BR-person](https://huggingface.co/rafola/BERT-base-pt-BR-person)

## Setup

```bash
pip install -r requirements.txt
cat exampletrain.txt > train.txt 
git clone https://huggingface.co/rafola/BERT-base-pt-BR-person model_output
```

## Generate 

*Tested with python 3.12.7*

```bash
python generate.py
```

## Test

```bash
python testmodel.py
```