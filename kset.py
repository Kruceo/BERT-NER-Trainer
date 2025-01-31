def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tokens = []
    labels = []
    temp_tokens = []
    temp_labels = []
    for line in lines:
        if line == '\n':
            tokens.append(temp_tokens)
            labels.append(temp_labels)
            temp_tokens = []
            temp_labels = []
            continue
        
        f = line.replace("\n","").split(" ")
        temp_tokens.append(f[0])
        temp_labels.append(f[1])
    tokens.append(temp_tokens)
    labels.append(temp_labels)
    return tokens, labels