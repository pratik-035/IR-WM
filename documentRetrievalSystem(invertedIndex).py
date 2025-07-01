
from collections import defaultdict 

class InvertedIndex: 
    def __init__(self, documents): 
        self.documents = documents 
        self.index = defaultdict(set) 
        
        #Dictionary to store the inverted index 
        self.build_index() 
        
    def build_index(self): 
        for doc_id, text in self.documents.items(): 
            #Convert text to lowercase and split into words 
            words = set(text.lower().split()) 
            #Map the word to the document ID 
            for word in words: 
                self.index[word].add(doc_id) 
                
    def search(self, query): 
        # Convert query to lowercase and split
        query_words = query.lower().split() 
        if not query_words: 
            #Return empty set if query is empty 
            return set() 
        
        #Retrieve the set of documents for each query word
        result_sets = [self.index[word] for word in query_words if word in self.index]
        
        if not result_sets:
            return set()
        
        #If none of the words are found, return an empty set 
        #Perform intersection to get documents containing all words 
        return set.intersection(*result_sets)   
    
    
#Sample Documents 
documents = { 
1: "Web Content extraction involves retrieving structured data", 
2: "Search engines use document indexing for efficient retrieval", 
3: "Document retrieval is important in web mining applications", 
4: "Indexing helps in retrieving relevant documents based on query terms" 
} 

#Create the inverted index 
index = InvertedIndex(documents) 

#Example search queries 
queries = ["retrieval", "document indexing", "web mining","structured data"] 

for query in queries:
    result = index.search(query)
    
    print(f"Query : '{query}' -> Documents : {sorted(result) if result else 'No matching Documents'}")