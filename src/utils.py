"""
Adapted from original implementation by yeonchang:
https://github.com/yeonchang/ASiNE

"""

import numpy as np
from typing import List, Dict, Tuple, Set, Union
import collections
import tqdm

# ---------------------------------------------------------------
# BFS TREE CONSTRUCTION (formerly construct_trees)
# ---------------------------------------------------------------

def make_bfs_trees(graph, nodes):
    """
    construct BFS trees with nodes
    config: (sub) graph, nodes according to the graph
    returns: trees (dictionary; node_id: [parent, child_0, child_1, ...])
    """
    trees = {}
    for root in tqdm.tqdm(nodes):  # Build BFS tree for each node in input list
        trees[root] = {}  # Initialize empty dict for current root node
        trees[root][root] = [root]  # Root node points to itself (BFS tree root characteristic)
        used_nodes = set()  # Track visited nodes to avoid duplicates
        queue = collections.deque([root])  # Double-ended queue (core BFS structure) initialized with root
        while len(queue) > 0:  # Process nodes until queue is empty (all reachable nodes visited)
            cur_node = queue.popleft()  # Get next node from left side of queue (FIFO)
            used_nodes.add(cur_node)  # Mark current node as visited

            for sub_node in graph[cur_node]:  # Iterate through all neighbors of current node
                if sub_node not in used_nodes:  # Only process unvisited neighbors
                    trees[root][cur_node].append(sub_node)  # Add neighbor as child of current node
                    trees[root][sub_node] = [cur_node]  # Create tree entry for neighbor with parent reference
                    queue.append(sub_node)
                    used_nodes.add(sub_node)
    return trees

# ---------------------------------------------------------------
# PATH → NODE PAIRS EXTRACTION (formerly get_node_pairs_from_path)
# ---------------------------------------------------------------

def extract_pairs_from_path(path,window_size):
    """
    Generate all node pairs within given window size from a path
    Example: path = [1, 0, 2, 4, 2], window_size = 2 -->
    node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
    Note: Excludes pairs beyond window size and self-loops
    Args:
        path: path from root to sampled node
    Returns:
        pairs: list of node pairs
    """
    path = path[:-1]  # Remove duplicate walk (source -> target)
    pairs = []
    for i in range(len(path)):
        center_node = path[i]
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(path))):
            if i == j or path[i] == path[j]:  # Exclude self-loops
                continue

            node = path[j]
            pairs.append([center_node, node])

    return pairs
# ---------------------------------------------------------------
# PATH → SIGNS (formerly get_node_pairs_sign_from_path)
# ---------------------------------------------------------------

def extract_signs_from_path(path,window_size):
    """
    Determine positive/negative nature of node pairs based on balance rule
    Args:
        path: path from root to sampled node
    Returns:
        pairs_sign: list indicating sign of each node pair
    """
    path = path[:-1]  # Remove duplicate walk
    pairs_sign = []
    for i in range(len(path)):
        center_node = path[i]
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(path))):
            if i == j or path[i] == path[j]:  # Exclude self-loops
                continue
            # Apply balance rule to determine pair sign
            if np.abs(i - j) % 2 == 1:  # Odd distance - negative
                pairs_sign.append([-1])
            else:  # Even distance - positive
                pairs_sign.append([1])

    return pairs_sign


def str_list_to_float(str_list: List[str]) -> List[float]:
    """Convert list of strings to list of floats.

    Args:
        str_list: List of string values to convert

    Returns:
        List of converted float values

    Raises:
        ValueError: If any element cannot be converted to float
    """
    try:
        return [float(item) for item in str_list]
    except ValueError as e:
        raise ValueError(f"Could not convert string to float: {e}")


def str_list_to_int(str_list: List[str]) -> List[int]:
    """Convert list of strings to list of integers.

    Args:
        str_list: List of string values to convert

    Returns:
        List of converted integer values

    Raises:
        ValueError: If any element cannot be converted to int
    """
    try:
        return [int(item) for item in str_list]
    except ValueError as e:
        raise ValueError(f"Could not convert string to int: {e}")


def read_edges(train_filename: str, test_filename: str = "") -> Tuple[
    Dict[int, List[int]], Dict[int, List[int]], int, List[int], List[int]]:
    """Read graph data from training and test files.

    Args:
        train_filename: Path to training edge file
        test_filename: Path to test edge file (optional)

    Returns:
        Tuple containing:
        - pos_graph: Positive adjacency list {node: [neighbors]}
        - neg_graph: Negative adjacency list {node: [neighbors]}
        - node_count: Total number of nodes
        - pos_nodes: Sorted list of positive nodes
        - neg_nodes: Sorted list of negative nodes
    """

    def add_edge_to_graph(node1: int, node2: int,
                          graph: Dict[int, List[int]],
                          positive: bool = True,
                          train: bool = True) -> None:
        """Helper function to add edges to graph."""
        if positive:
            pos_nodes.add(node1)
            pos_nodes.add(node2)
        else:
            neg_nodes.add(node1)
            neg_nodes.add(node2)

        # Initialize adjacency lists if needed
        graph.setdefault(node1, [])
        graph.setdefault(node2, [])

        # For training data, add bidirectional connections
        if train:
            graph[node1].append(node2)
            graph[node2].append(node1)

    def count_edge_types(edges: List[List[Union[int, float]]]) -> Tuple[int, int]:
        """Count positive and negative edges in edge list."""
        pos = sum(1 for edge in edges if edge[2] > 0)
        neg = sum(1 for edge in edges if edge[2] < 0)
        return pos, neg

    pos_graph, neg_graph = {}, {}
    pos_nodes, neg_nodes = set(), set()

    # Read and process training edges
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename else []

    # Calculate and print dataset statistics
    total_edges = len(train_edges) + len(test_edges)
    print(f"Train ratio: {len(train_edges) / total_edges:.2f}")

    train_pos, train_neg = count_edge_types(train_edges)
    test_pos, test_neg = count_edge_types(test_edges)

    print(f"Positive edges in Train: {train_pos / len(train_edges):.2f}")
    print(f"Positive edges in Test: {test_pos / len(test_edges):.2f}")

    # Process training edges
    print("Processing training edges...")
    for src, dst, weight in train_edges:
        if weight == 1:
            add_edge_to_graph(src, dst, pos_graph)
        elif weight == -1:
            add_edge_to_graph(src, dst, neg_graph, positive=False)

    # Process test edges (no bidirectional connections)
    if test_edges:
        print("Processing test edges...")
        for src, dst, weight in test_edges:
            if weight == 1:
                add_edge_to_graph(src, dst, pos_graph, train=False)
            elif weight == -1:
                add_edge_to_graph(src, dst, neg_graph, positive=False, train=False)

    # Prepare final outputs
    all_nodes = pos_nodes | neg_nodes
    node_count = max(all_nodes) + 1 if all_nodes else 0

    return (
        pos_graph,
        neg_graph,
        node_count,
        sorted(pos_nodes),
        sorted(neg_nodes)
    )


def read_edges_from_file(filename: str) -> List[List[int]]:
    """Read edges from file and parse as list of integer tuples.

    Args:
        filename: Path to edge file

    Returns:
        List of edges as [src, dst, weight] tuples

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(filename, "r") as f:
            return [str_list_to_int(line.split()) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"Edge file not found: {filename}")


def read_embeddings(filename: str, n_node: int, n_embed: int) -> np.ndarray:
    """Read pretrained node embeddings from file.

    Args:
        filename: Path to embedding file
        n_node: Number of nodes expected
        n_embed: Embedding dimension

    Returns:
        Embedding matrix of shape (n_node, n_embed)
    """
    try:
        with open(filename, "r") as f:
            embedding_matrix = np.random.rand(n_node, n_embed)
            for line in f.readlines()[1:]:  # Skip header
                parts = line.split()
                node_id = int(parts[0])
                embedding_matrix[node_id] = str_list_to_float(parts[1:])
            return embedding_matrix
    except (FileNotFoundError, IndexError) as e:
        raise ValueError(f"Error reading embeddings: {e}")


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for input array.

    Args:
        x: Input array

    Returns:
        Softmax normalized array
    """
    e_x = np.exp(x - np.max(x))  # Numerical stability
    return e_x / e_x.sum(axis=0)


def divide_chunks(iterable: List, chunk_size: int):
    """Yield successive chunks from iterable.

    Args:
        iterable: Input list to chunk
        chunk_size: Size of each chunk

    Yields:
        Chunks of the original list
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def aggregate_link_emb(link_method: str,
                       emb1: np.ndarray,
                       emb2: np.ndarray) -> np.ndarray:
    """Aggregate two node embeddings using specified method.

    Args:
        link_method: One of:
            - 'weight_l1': L1 distance
            - 'weight_l2': Squared L2 distance  
            - 'concatenation': Vector concatenation
            - 'average': Element-wise average
            - 'Hadamard': Element-wise product
            - 'addition': Element-wise sum
        emb1: First node embedding
        emb2: Second node embedding

    Returns:
        Aggregated embedding vector

    Raises:
        ValueError: For unknown aggregation methods
    """
    if link_method == "weight_l1":
        return np.abs(emb1 - emb2)
    elif link_method == "weight_l2":
        return (emb1 - emb2) ** 2
    elif link_method == "concatenation":
        return np.concatenate([emb1, emb2])
    elif link_method == "average":
        return (emb1 + emb2) / 2
    elif link_method == "Hadamard":
        return emb1 * emb2
    elif link_method == "addition":
        return emb1 + emb2
    else:
        raise ValueError(f"Unknown link aggregation method: {link_method}")