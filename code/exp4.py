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
