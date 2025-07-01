
from collections import defaultdict 
import numpy as np

class InvertedIndex: 
    def __init__(self, documents): 
        """
        Initializes the inverted index with a given set of documents. 
        :param documents: A dictionary where keys are document IDs and 
        values are document text. 
        """
        
        self.documents = documents 
        self.index = defaultdict(set) 
        
        self.vocabulary = set() # store unique words for spell correction 
        
        self.build_index()
        
    def build_index(self):
        """
        Builds the inverted index from the provided documents.
        """
        
        for doc_id, text in self.documents.items():
            words = set(text.lower().split()) # convert text to lowercase and split into words 
            
            self.vocabulary.update(words)
            
            for word in words:
                self.index[word].add(doc_id) # map the word to the documet ID 
                
    
    def search(self, query):
        
        """
        Searches for documents containing words in the query.
        :param query: A user query string containing potentially 
        misspelled words. 
        :return: A set of document IDs matching the corrected query words.
        """
        
        corrected_words = [self.corrected_spelling(word) for word in query.lower().split()]
        
        print(f"Corrected Query : {' '.join(corrected_words)}")
        
        result_sets = [self.index[word] for word in corrected_words if word in self.index]
        
        return set.intersection(*result_sets) if result_sets else set() # find common documents 
    
    
    def corrected_spelling(self, word): 
        """
        Corrects the spelling of a given word using the Levenshtein 
        Distance. 
        :param word: The potentially misspelled word.
        :return: The closest matching word from the vocabulary. 
        """
        
        min_distance = float('inf')
        best_match = word # default to original word if no close match is found 
        
        
        for vocab_word in self.vocabulary: 
            
            distance = self.levenshtein_distance(word, vocab_word)
            
            if distance < min_distance:
                min_distance = distance
                best_match = vocab_word # update best match if a close word is found 
                
        return best_match
    
    
    def levenshtein_distance(self, s1, s2): 
        """
        Computes the Levenshtein Distance between two words. 
 
        :param s1: First word. 
 
        :param s2: Second word. 
 
        :return: The edit distance between the two words. 
        """
        
        len_s1, len_s2 = len(s1), len(s2)
        dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)
        
        
        for i in range(len_s1 + 1):
            for j in range(len_s2 + 1):
                
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                       dp[i][j - 1], # Insertion
                                       dp[i - 1][j - 1]) # Substituion 
                    
        return dp[len_s1][len_s2]
        
        
# Sample documents 

documents = { 
1: "Web content extraction involves retrieving structured data", 
2: "Search engines use document indexing for efficient retrieval", 
3: "Document retrieval is important in web mining applications", 
4: "Indexing helps in retrieving relevant documents based on query terms" 
}

# Create the inverted index 
index = InvertedIndex(documents)

# Example search queries with misspelled words 
 
queries = ["retrievel", "documnt indexing", "web minng", "strctured data"]
print("\n=== Spelling Correction and Document Retrieval ===") 

for query in queries:
    
    result = index.search(query)
    
    print(f"Query : '{query}' -> Corrected Documents: {sorted(result) if result else 'No matching Documents'}")