import nltk
nltk.download('punkt')  # Download the tokenizer model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt

# nltk.download('wordnet')


message ="""
Hi. Looking for some advice on the order routing feature in Shopify plus. In the past this was never an issue but since the implementation of this feature it seems to be causing a lot of issues on our side. 

Example. We have branch A with 5 in stock and branch B with 1 in stock. If I order 6 items, the system should allocate them to the 2 branches for fulfillment & delivery but at the moment Shopify is allocating 6 to branch A. We have reverted to the default, changed different ordering of the feature as well. It seems they do not look at stock anymore with this. Even though it’s limiting the splitting of the fulfillments, it should still pay attention to the stock levels at that location. 

Hope someone can advise if they experience the same. Thanks.
"""



tokens = word_tokenize(message)
print(tokens)  # ['Hello', ',', 'world', '!']



stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]


print(filtered_tokens)


lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]


filtered_words = [x.lower() for x in filtered_tokens if re.sub(r'[^a-zA-Z\s]', '', x) != '']
print(filtered_words)

N=8
# Get word frequency
freq = Counter(filtered_words)

# Select top N words
common = freq.most_common(N)
words, counts = zip(*common)

# Plot
plt.figure(figsize=(10,5))
plt.bar(words, counts, color='blue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title(f'Top {N} Words')
plt.show()
