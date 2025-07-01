
import numpy as np 

def initialize_pagerank(graph):
    """
    Initialize the PageRank values for all nodes in the graph. 
    graph : Dictionary represnting the adjacency list of the web graph.
    return: Dictionary of initial PageRank values 
    """

    num_nodes = len(graph)

    return {node: 1 / num_nodes for node in graph} # All nodes start with equal probability

def compute_pagerank(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    """
    Computes the PageRank scores iteratively until convergence of the max interation is reached. 
    :graph : Dictionary representing the adjacency list of the web graph.
    :damping_factor : Probability of following a link (default 0.85)
    :max_iterations : Maximum number of iterations
    :tolerance : Convergence threshold
    :return : Dictionary of final PageRank values.
    """

    num_nodes = len(graph)

    pagerank = initialize_pagerank(graph) # Initialize PR values
    
    new_pagerank = pagerank.copy()

    for _ in range(max_iterations):
        for node in graph:
            # Calculate new PR value using the PageRank formula

            new_pagerank[node] = (1 - damping_factor) / num_nodes + damping_factor * sum (
                pagerank[incoming] / len(graph[incoming]) for incoming in graph if node in graph[incoming]
            )


            # Check convergence (if the change in values is below tolerance, stop iterating)

        if all(abs(new_pagerank[node] - pagerank[node]) < tolerance for node in pagerank):
            break

        pagerank = new_pagerank.copy() # Update PR values

    return pagerank


# -------------- EXAMPLE DATASET (Web Graph) ------------------------

# Representaion of a small web graph as an adjacency list

web_graph =  { 
    'A': {'B', 'C'},
    'B': {'C'},
    'C': {'A'},
    'D':{'C'}
}

# Compute PageRank
pagerank_scores = compute_pagerank(web_graph)


# Display Results

print("\n=== PageRank Scores ===")

for page, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True):

    print(f"Page {page} : {score:.6f}")