from nltk.tokenize import word_tokenize

class Tokenizer:
    @staticmethod
    def tokenize(text):
        """
        Tokenizes the input text by converting it to lowercase and then
        splitting it into words using NLTK's word_tokenize method.
        """
        tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
        return tokens