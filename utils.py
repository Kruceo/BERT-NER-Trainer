def generate_label_to_id(labels):
    mapping = []
    out = {}
    for label in labels:
        if not mapping.__contains__(label):
            out[label] = len(out)
            mapping.append(label)
    return out

def save_label_to_id(filepath,obj):
    with open(filepath,"w") as f:
        content = ""
        for key in obj:
            content += f"{key}={obj[key]}\n"
        f.write(content[:len(content)-1])
            
            
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def pretty_print_results(results):
    output = []
    for r in results:
        output.append([r["entity"],f"{r["word"]}"])

    total = ""    
    for word in output:
        color = bcolors.ENDC
        if word[0] == "LABEL_1":
            color = "color2"
        if word[0] == "LABEL_0":
            color = "color1"
        total += (" " if not word[1].__contains__("##") else "") +  (f"<{color}>" if word[0]!="LABEL_2" else "") + word[1].replace("##","") +  (bcolors.ENDC if word[0]!="LABEL_2" else "")
    
    result = total[1:].replace("¶","\n").replace(" , ",", ").replace(" . ",". ").replace(" ? ","? ").replace(" : ",": ").replace(" r $ "," R$ ").replace(" !","!")
    result = result.replace("<color1>",bcolors.OKCYAN)
    result = result.replace("<color2>",bcolors.OKBLUE)
    print(result)