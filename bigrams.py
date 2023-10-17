import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams as NlBigrams
from collections import Counter
import matplotlib.pyplot as plt

# Download required datasets
nltk.download('punkt')
nltk.download('stopwords')


def plot_top_n_bigrams(text, N=10):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert to lowercase and remove stopwords
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Generate bigrams
    bigram_list = list(NlBigrams(tokens))

    # Get bigram frequency
    freq = Counter(bigram_list)

    # Select top N bigrams
    common = freq.most_common(N)
    bigrams, counts = zip(*common)
    bigrams = [" ".join(bigram) for bigram in bigrams]  # Convert tuple to string for display

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(bigrams, counts, color='green')
    plt.xlabel('Bigrams')
    plt.ylabel('Frequency')
    plt.title(f'Top {N} Bigrams')
    plt.xticks(rotation=45)  # Rotate labels for better visibility
    plt.tight_layout()
    plt.show()

# Example usage:
text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."""
plot_top_n_bigrams(text, N=4)
