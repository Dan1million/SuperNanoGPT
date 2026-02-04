class Tokenizer():
    """
        Basic tokenizer for encoding and decoding tokens based on the unique
        characters in a text
    """

    def __init__(self, text=None, vocab=None):
        """
            Initializes the tokenizer

            Args:
                text string: the data set text for building the vocabulary
                vocab list: pre-built vocabulary (optional)
        """
        if vocab is not None:
            self.chars = vocab
        elif text is not None:
            self.chars = sorted(list(set(text)))
        else:
            raise ValueError("Either text or vocab must be provided")
        
        # Handle rare tokens by making them <UNK>
        if '<UNK>' not in self.chars:
            self.chars.append('<UNK>')
        self.unk_token = '<UNK>'

    def encode(self, chunk):
        """
            Encodes the unique characters to a list of integers representing
            the individual characters. Unknown characters are mapped to <UNK> token.

            Args:
                chunk string: the string to encode
        """
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        unk_idx = stoi[self.unk_token]
        return [stoi.get(c, unk_idx) for c in chunk]

    def decode(self, list):
        """
            Decodes the integer list into a string of the corresponding characters

            Args:
                list int[]: list of integers representing characters
        """
        itos = { i:ch for i,ch in enumerate(self.chars) }
        return ''.join([itos[i] for i in list])
    
    def vocab_size(self):
        return len(self.chars)
    
    def save_vocab(self, filepath):
        """
            Saves the vocabulary to a file
            
            Args:
                filepath string: path to save the vocabulary
        """
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.chars, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_vocab(filepath):
        """
            Loads a tokenizer from a saved vocabulary file
            
            Args:
                filepath string: path to the vocabulary file
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return Tokenizer(vocab=vocab)
