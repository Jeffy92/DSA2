import random
import networkx
from typing import List, Tuple, Optional

# Visualization (empty board and final board)
def visualize_board(size: int, queens: List[int]) -> None:
    """Prints a size x size board. '.' = empty, 'Q' = queen.. queens is a list where index is row and value is column."""
    board = [['.' for _ in range(size)] for _ in range(size)]
    
    #place queens
    for row, col in enumerate(queens):
        board[row][col] = 'Q'
    for row in board:
        print(' '.join(row))
        
    #column header
    print(" " + " ".join(f"{c:2d}" for c in range(size)))
    print(" " + "--" * size)
    
    #numbered rows
    for r in range(size):
        print(f"{r:2d}| " + " ".join(f"{cell:2s}" for cell in board[r]))
    print() 
def show_empty_board(size: int) -> None:
    """Prints an empty size x size board."""
    visualize_board(size, [])

def is_safe(queens: List[int], row: int, col: int) -> bool:
    """Check if placing a queen at (row, col) is safe given existing queens."""
    for existing_col, existing_row in enumerate(queens):
        if existing_row == row or abs(existing_row - row) == abs(existing_col - col):
            return False
    return True
    
#Las Vegas Algorithm for N-Queens
def nQueensLasVegas(size: int) -> Tuple[bool, List[int]]:
        """Las Vegas algorithm: Place queens column by column. At each column randomly select a safe row. Failure if no safe row."""
        queens = []
        for col in range(size):
            safe_rows = [r for r in range(size) if is_safe(queens, r, col)]
            if not safe_rows:
                return False, [] #unsuccessful
            queens.append(random.choice(safe_rows))
        return True, queens #successful
# Backtracking Algorithm for N-Queens
def nQueensBacktracking(size: int) -> Tuple[bool, List[List[int]]]:
    """Backtracking algorithm: Recursively place queens column by column. Try rows in order; backtrack on failure."""
    solutions = []
    def backtrack(col: int, queens: List[int]) -> bool:
        if col == size:
            solutions.append(queens[:])
            return True
        for row in range(size):
            if is_safe(queens, row, col):
                queens.append(row)
                backtrack(col + 1, queens)
                queens.pop()
        return False

    backtrack(0, [])
    return (len(solutions) > 0, solutions[0] if solutions else [])


def run_n_equals_4():
    size = 4
    print(f"Running N-Queens for N={size}\n")
    
    print("Empty Board:")
    show_empty_board(size)
    
    print("Las Vegas Algorithm:")
    success, queens = nQueensLasVegas(size)
    if success:
        visualize_board(size, queens)
    else:
        print("No solution found.")
    
    print("Backtracking Algorithm:")
    success, queens = nQueensBacktracking(size)
    if success:
        visualize_board(size, queens)
    else:
        print("No solution found.")
if __name__ == "__main__":
        run_n_equals_4()

def run_n_equals_4():
        n = 4
        print(f"Running N-Queens for N={n}\n")
        
        print("Empty Board:")
        show_empty_board(n) 
        success = True 
        queens = [[0, 2], [1, 0], [2, 3], [3, 1]]
        
        print(f"Result: {'success' if success else 'failure'}")
        print("Final Board:")
        visualize_board(n, queens)
        
        if __name__ == "__main__":
            run_n_equals_4()

#Backtracking version 2 
def nQueensBacktrackingVersion2(size: int, startingPostion: tuple[int, int]) -> Tuple[bool, List[List[int]]]:
    start_row, start_col = startingPostion
    queens: List[List[int]] = []
    solutions = []
    if not (0 <= start_row < size and 0 <= start_col < size):
        return False, []  # Invalid starting position
    
    def backtrack(col: int) -> bool:
        if col == size:
            solutions.append(queens[:])
            return True
        if col == start_col:
            return backtrack(col+1)
        for row in range(size):
            if is_safe(queens, row, col):
                queens.append(row)
                backtrack(col + 1)
                queens.pop()
        return False
    
    success = backtrack (0)
    return (success, solutions[0] if solutions else [])

#Input manager (user input, unexpected input handling + exit program)
def read_int(prompt: str, min_value: Optional[int] = None) -> Optional[int]:
    """Reads an integer from user. User can type 'exit' or 'quit' to exit program.
    Return None if user chooses to exit."""
    exit_words = {"exit", "quit"}
    while True: 
        user_input = input(prompt).strip().lower()
        if user_input in exit_words:
            return None
        try:
            value = int(user_input)
            if min_value is not None and value < min_value:
                print(f"Please enter an integer >= {min_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer or type 'exit' or 'quit'.")

    while True:
        def read_choice(prompt: str, choices: List[str]) -> Optional[str]:
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in {"exit", "quit"}:
                    return None
                if user_input in choices:
                    return user_input
                print("Invalid input. Please enter a valid choice or type 'exit' or 'quit'.")
                
#Main program
def main() -> None:
    print("N-Queens Solver")
    print("Type 'exit' or 'quit' at any prompt to exit the program.\n")
    while True:
        size = read_int("Enter the size of the board (N >= 1) or type 'exit' to quit: ", min_value=1)
        if size is None:
            print("Exiting program.")
            return
        
        print("Choose algorithm:")
        print("1. Las Vegas")
        print("2. Backtracking")
        print("3. Backtracking Version 2 (with starting position)")
        choice = read_choice("Enter choice (1/2/3) or type 'exit' to quit: ", ["1", "2", "3"])
        if choice is None:
            print("Exiting program.")
            return

        if choice == "1":
            success, queens = nQueensLasVegas(size)
        elif choice == "2":
            success, queens = nQueensBacktracking(size)
        else:  # choice == "3"
            # Get starting row from user
            start_row = read_int("Enter starting row (0-indexed): ", min_value=0)
            if start_row is None:
                print("Exiting program.")
                return
            # Get starting column from user
            start_col = read_int("Enter starting column (0-indexed): ", min_value=0)
            if start_col is None:
                print("Exiting program.")
                return
            if start_row >= size or start_col >= size:
                success, queens = False, []
            else:
                success, queens = nQueensBacktrackingVersion2(size, (start_row, start_col))
        
        print(f"Result: {'success' if success else 'failure'}")
        print("Final Board:")
        visualize_board(size, queens)
        again = read_choice("Do you want to solve another? (yes/no): ", ["yes", "no"])
        if again is None or again == "no":
            print("Exiting program.")
            return
            
if __name__ == "__main__":
    main()




import networkx as nx
import matplotlib.pyplot as mpl 
# MST algorithm Kruskal's algorithm (union-find)

class UnionFind:
    def __init__ (self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
        
    def find(self, item):
        #find root of item with path compression
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    def union(self, set1, set2):
        #union sets by rank
        root1 = self.find(set1)
        root2 = self.find(set2)
        
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False
                

def MyMinimumSpanningTree(Graph) -> nx.Graph:
    """Kruskal's algorithm to find Minimum Spanning Tree of a connected, undirected, weighted graph."""
    uf = UnionFind(Graph.nodes())
    mst = nx.Graph()
    mst.add_nodes_from(Graph.nodes(data=True))
    #sort edges by weight
    edges = sorted(Graph.edges(data=True), key=lambda x: x[2]['weight'])

    print("\n--- Building Minimum Spanning Tree using Kruskal's Algorithm ---\n")
    total_weight = 0
    edges_added = 0
    target_edges = Graph.number_of_nodes() - 1

    for u, v, data in edges:
        w = data["weight"]
        print(f"Considering edge ({u}, {v}) with weight {w}...", end="")
        
        if uf.union(u, v):
            mst.add_edge(u, v, weight=w)
            total_weight += w
            edges_added += 1
            print(f" added to MST. Current total weight: {total_weight}")
        else:
            print(f" rejected to avoid cycle.")
        
        #stop if we have enough edges
        if edges_added == target_edges:
            break
    print(f"Minimum Spanning Tree completed with total weight: {total_weight}\n")
    return mst



# test graphs

import networkx as nx
import matplotlib.pyplot as plt 
def make_graph_1():
    """Create and return graph with 6 nodes and 10 edges."""
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 4),
        (0, 2, 3),
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 4),
        (3, 4, 2),
        (4, 5, 6),
        (3, 5, 3),
        (1, 5, 7),
        (2, 4, 5)
    ])
    return G

def draw_graph(G):
    """Draw the graph with weights."""
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Connect graph with 6 nodes and 10 edges")
    plt.show()  
def main():
    """Main function to create and draw the graph."""
    G = make_graph_1()
    draw_graph(G)

if __name__ == "__main__":
    main() 
    
    def make_graph_3():
    """Create and return graph with 5 nodes and 8 edges."""
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 3),
        (0, 2, 2),
        (1, 2, 4),
        (1, 3, 5),
        (2, 3, 1),
        (3, 4, 6),
        (1, 4, 7),
        (2, 4, 8)
    ])
    return G

def draw_graph_3(G):
    """Draw the graph with weights."""
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='orange', node_size=500, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Connect graph with 5 nodes and 8 edges")
    plt.show()
    
def main_3():
    """Main function to create and draw the graph."""
    G = make_graph_3()
    draw_graph_3(G)
if __name__ == "__main__":
    main_3()
    





def findMostFrequentWord(a1, a2):
    freq = {}
    for word in a1:
        freq[word] = freq.get(word, 0) + 1

    for word in a2:
        if word in freq:
            freq[word] = -1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_words) < 2:
        return ""
    
    return sorted_words[1][0]

def main():
    print("== part a) example ==")
    a1 = ["apple", "banana", "apple", "cat", "banana", "dog", "banana"]
    a2 = ["dog"] 
    print("Input list 1:", a1)
    print("Input list 2:", a2)
    
    result = findMostFrequentWord(a1, a2)
    print("second most frequent word:", result)


#helper recursive function
def isPalindromeRecursive(s: str, left: int, right: int) -> bool:
    if left >= right:
        return True
    if s[left] != s[right]:
        return False
    return isPalindromeRecursive(s, left + 1, right - 1)      

def isPalindrome(s):
    """Check if string s is a palindrome using recursion."""
    return isPalindromeRecursive(s, 0, len(s) - 1)

if __name__ == "__main__":
    s = "racecar"
    if isPalindrome(s):
        print("true")
    else:
        print("false")