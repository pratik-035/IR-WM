
from collections import defaultdict 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 

class BooleanRetrieval: 
    def __init__(self, documents): 
        """
        nitializes the inverted index for Boolean Retrieval. 
 
        :param documents: A dictionary where keys are document IDs and 
        values are document text. 
 
        """ 
        
        self.documents = documents 
 
        self.inverted_index = defaultdict(set) #Dictionary to store the inverted index 
 
        self.build_index() 
        
    def build_index(self): 
        """
        "Builds the inverted index from the provided documents."
        """
        
        for doc_id, text in self.documents.items():
            words = set(text.lower().split()) 
            
            for word in words: 
                self.inverted_index[word].add(doc_id) 
                
                
    def boolean_search(self, query): 
        """
        Performs a Boolean search with AND, OR, and NOT operations. 
 
        :param query: A Boolean search query (e.g., "retrieval AND 
        document"). 
 
        :return: A set of document IDs matching the query.
        """
        
        terms = query.lower().split() 
        
        result_set = set(self.documents.keys())
        
        
        operation = "AND" # defaulr operation 
        for term in terms:
            
            if term == "and":
                operation = "AND"
            elif term == "or":
                operation = "OR"
            elif term == "not":
                operation = "NOT"
            else:
                if operation == "AND":
                    result_set &= self.inverted_index.get(term, set())
                elif operation == "OR":
                    result_set |= self.inverted_index.get(term, set())
                elif operation =="NOT":
                    result_set -=  self.inverted_index.get(term, set())
        
        return result_set
    
    
class VectorSpaceRetrieval:
        
        def __init__(self, documents):
            """
            Initializes the TF-IDF vectorization for document retrieval. 
 
            :param documents: A dictionary where keys are document IDs and 
            values are document text. 
            """
            
            self.documents = documents
            self.vectorizer = TfidfVectorizer()
            self.doc_ids = list(documents.keys())
            self.doc_vectors = self.vectorizer.fit_transform(documents.values())
            
            
        def vector_search(self, query):
            """
            performs a vector space search using cosine similarity. 
 
            :param query: A text query. 
 
            :return: A list of (document ID, similarity score) sorted by 
            relevance.
            """
            
            query_vector = self.vectorizer.transform([query]) # convert query to tfidf vector 
            
            similarities = np.dot(self.doc_vectors, query_vector.T).toarray().flatten() # compute cosine similarity
            
            ranked_results = sorted(zip(self.doc_ids, similarities), key=lambda x: x[1], reverse=True)
            
            
            return ranked_results
        
# Sample documents 

documents = { 
1: "Web content extraction involves retrieving structured data", 
2: "Search engines use document indexing for efficient retrieval", 
3: "Document retrieval is important in web mining applications", 
4: "Indexing helps in retrieving relevant documents based on query terms" 
} 

# boolean retireval system 

boolean_index = BooleanRetrieval(documents)
boolean_queries = ["retrieval AND document", "document OR indexing", "retrieval NOT indexing"]

for query in boolean_queries:
    
    result = boolean_index.boolean_search(query)
    print(f"Query : '{query}' -> Documents : {sorted(result) if result else 'No matching documents'}")
    
# Vector Space Retrieval System 
vector_index = VectorSpaceRetrieval(documents)
vector_queries = ["document retrieval", "web mining", "structured data"]

print("\n ==== Vector Space Model Results ===== ")

for query in vector_queries:
    
    result = vector_index.vector_search(query)
    print(f"Query : '{query}' -> Ranked Documents : {[(doc, round(score, 4)) for doc, score in result if score > 0]}")