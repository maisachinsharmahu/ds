# Experiment 4: Python for the Analysis of Data Structures

> **File:** `exp4.py`  |  **Tools:** Python 3, collections

## 🎯 Aim

To implement and analyse common data structures in Python.

## 📘 Theory

Beyond the built-in types, many problems need structures written by the programmer. A linked list keeps items in nodes that point to the next node; a binary search tree keeps smaller values on the left and larger on the right, so that an inorder walk gives sorted values; a graph stored as a dictionary of neighbours can be walked breadth first (with a queue) or depth first (with recursion). A topological order puts every node before the nodes it points to.

A stack evaluates postfix expressions, and memoisation (a dictionary of earlier results) turns the exponential Fibonacci recursion into a linear one.

- **Linked list**: Chain of nodes.
- **BST**: Tree with ordered left and right children.
- **BFS / DFS**: Breadth first and depth first search.
- **Memoisation**: Storing results of earlier calls.

## 🪜 Algorithm / Steps

1. Build a linked list with push and reverse it.
2. Insert eight values into a binary search tree, walk it inorder and measure the height.
3. Traverse a six node graph with BFS and DFS, find a topological order and the number of steps from A.
4. Evaluate a postfix expression with a stack and compute Fibonacci(40) with memoisation.
5. Group anagrams with a dictionary and add rows and columns of a matrix.

## 💻 Code

```python
from collections import deque


class Node:
    def __init__(self, v):
        self.v, self.next = v, None


class LinkedList:
    def __init__(self):
        self.head, self.n = None, 0

    def push(self, v):
        x = Node(v)
        x.next = self.head
        self.head = x
        self.n += 1

    def reverse(self):
        prev, cur = None, self.head
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        self.head = prev

    def items(self):
        out, c = [], self.head
        while c:
            out.append(c.v)
            c = c.next
        return out


class BST:
    def __init__(self):
        self.root = None

    def insert(self, v):
        if self.root is None:
            self.root = [v, None, None]
            return
        c = self.root
        while True:
            i = 1 if v < c[0] else 2
            if c[i] is None:
                c[i] = [v, None, None]
                return
            c = c[i]

    def inorder(self, c="root"):
        c = self.root if c == "root" else c
        return (
            []
            if c is None
            else self.inorder(c[1]) + [c[0]] + self.inorder(c[2])
        )

    def height(self, c="root"):
        c = self.root if c == "root" else c
        return (
            0
            if c is None
            else 1 + max(self.height(c[1]), self.height(c[2]))
        )


ll = LinkedList()
for v in [5, 3, 8, 1]:
    ll.push(v)

print("linked list:", ll.items(), "length", ll.n)

ll.reverse()

print("reversed   :", ll.items())

t = BST()
for v in [50, 30, 70, 20, 40, 60, 80, 35]:
    t.insert(v)

print("\nBST inorder:", t.inorder(), "| height", t.height())

graph = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D", "E"],
    "D": ["F"],
    "E": ["F"],
    "F": [],
}


def bfs(s):
    seen, q, order = {s}, deque([s]), []
    while q:
        u = q.popleft()
        order.append(u)
        for w in graph[u]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return order


def dfs(u, seen=None):
    seen = seen if seen is not None else []
    seen.append(u)
    for w in graph[u]:
        if w not in seen:
            dfs(w, seen)
    return seen


print("\ngraph BFS:", bfs("A"), "| DFS:", dfs("A"))

indeg = {u: 0 for u in graph}
for u in graph:
    for w in graph[u]:
        indeg[w] += 1
order, q = [], deque([u for u in graph if indeg[u] == 0])
while q:
    u = q.popleft()
    order.append(u)
    for w in graph[u]:
        indeg[w] -= 1
        if indeg[w] == 0:
            q.append(w)

print("topological order:", order)

dist = {u: float("inf") for u in graph}
dist["A"] = 0
for u in bfs("A"):
    for w in graph[u]:
        dist[w] = min(dist[w], dist[u] + 1)

print("steps from A:", dist)

stack = []
for tok in "3 4 + 2 *".split():
    (
        stack.append(int(tok))
        if tok.isdigit()
        else stack.append(
            {"+": lambda a, b: a + b, "*": lambda a, b: a * b}[tok](
                stack.pop(-2), stack.pop()
            )
        )
    )

print("\npostfix 3 4 + 2 * =", stack[0])


def fib(n, memo={}):
    if n < 2:
        return n
    if n not in memo:
        memo[n] = fib(n - 1) + fib(n - 2)
    return memo[n]


print("memoised fibonacci(40) =", fib(40))

words = ["tea", "eat", "tan", "nat", "bat", "ate"]
groups = {}
for w in words:
    groups.setdefault("".join(sorted(w)), []).append(w)

print("anagram groups:", list(groups.values()))

matrix = [[2, 0, 1], [1, 3, 2], [0, 1, 4]]

print(
    "row sums",
    [sum(r) for r in matrix],
    "| column sums",
    [sum(c) for c in zip(*matrix)],
    "| trace",
    sum(matrix[i][i] for i in range(3)),
)
```

## 🖥️ Output

```text
linked list: [1, 8, 3, 5] length 4
reversed   : [5, 3, 8, 1]

BST inorder: [20, 30, 35, 40, 50, 60, 70, 80] | height 4

graph BFS: ['A', 'B', 'C', 'D', 'E', 'F'] | DFS: ['A', 'B', 'D', 'F', 'C', 'E']
topological order: ['A', 'B', 'C', 'D', 'E', 'F']
steps from A: {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 3}

postfix 3 4 + 2 * = 14
memoised fibonacci(40) = 102334155
anagram groups: [['tea', 'eat', 'ate'], ['tan', 'nat'], ['bat']]
row sums [3, 6, 5] | column sums [3, 4, 7] | trace 9
```

## ✅ Result

The linked list [1, 8, 3, 5] reversed to [5, 3, 8, 1]. The tree gave the sorted values [20, 30, 35, 40, 50, 60, 70, 80] with height 4. BFS visits A, B, C, D, E, F and DFS A, B, D, F, C, E; the topological order is A to F and the number of steps from A are 0, 1, 1, 2, 2, 3. The postfix expression 3 4 + 2 * is 14 and Fibonacci(40) = 102334155 was found instantly. The anagram groups are [tea, eat, ate], [tan, nat] and [bat]; the matrix has the row sums 3, 6, 5, the column sums 3, 4, 7 and the trace 9.

## 🎤 Viva Questions

**Q. What is the difference between a list and a tuple?**  
A. A list can be changed, a tuple cannot.

**Q. Why is membership in a set fast?**  
A. A set uses hashing, so the lookup time is almost constant.

**Q. What is a dictionary?**  
A. A collection of key and value pairs with fast lookup by key.

**Q. What are a stack and a queue?**  
A. A stack is last in, first out; a queue is first in, first out (deque is used for it).

**Q. What does Counter do?**  
A. It counts how often each item occurs.

---
[⬅ Back to index](README.md)
