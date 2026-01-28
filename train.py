with open('input.txt', 'r', encoding='utf-8') as f :
    text = f.read()

chars = sorted(list(set(text))) # create the vocabulary by parsing the unique characters
vocab_size = len(chars)
