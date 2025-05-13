import logging
import datetime
import queue

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logger.addHandler(console)


class HOC:
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # 指向根
        self.edgeTs = {}  # 根出发
        self.node2leaf = {}
        self.node2ancestor = {}
        self.node2belong = {}

    def init_node(self, u):
        self.nodes.add(u)
        if u not in self.edges:
            self.edges[u] = set()
            self.edgeTs[u] = set()

    def add_edge(self, u, v):
        if u not in self.nodes:
            self.init_node(u)
        if v not in self.nodes:
            self.init_node(v)
        self.edges[u].add(v)
        self.edgeTs[v].add(u)

    def remove_edge(self, u, v):
        if u in self.edges and v in self.edges[u]:
            self.edges[u].remove(v)
            self.edgeTs[v].remove(u)

    def get_parents(self, u):
        if u in self.nodes:
            parents = self.edgeTs[u]
        else:
            raise ValueError("node {} not in DAG".format(u))
        return parents

    def get_children(self, u):
        if u in self.nodes:
            children = self.edges[u]
        else:
            raise ValueError("node {} not in DAG".format(u))
        return children

    def topological_sort(self):
        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for neighbor in self.edges[node]:
                in_degree[neighbor] += 1
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in self.edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            # print(queue, node)
        if len(result) != len(self.nodes):
            raise ValueError("The graph contains a cycle")
        return result

    def init_ancestors(self):
        for u in self.nodes:
            self.node2ancestor[u] = set()
        # print(self.topological_sort())
        for u in self.topological_sort():
            for v in self.edges[u]:
                self.node2ancestor[v] |= (self.node2ancestor[u] | {u})
            # print(self.node2ancestor)

    def get_ancestors(self, u):
        if u in self.nodes:
            ancestors = self.node2ancestor[u]
        else:
            raise ValueError("node {} not in DAG".format(u))
        return ancestors

    def get_mca(self, u, v):
        fu = self.get_ancestors(u)
        fv = self.get_ancestors(v)
        mca = set()
        S = fu & fv
        for node in S:
            if not any(neighbor in S for neighbor in self.edges[node]):
                mca.add(node)
        return mca

    def get_node2leaf(self, u):
        children = self.get_children(u)
        self.node2leaf[u] = set()
        if len(children) == 0:
            self.node2leaf[u].add(u)
        else:
            for child in children:
                self.get_node2leaf(child)
                self.node2leaf[u] = self.node2leaf[u] | self.node2leaf[child]

    def get_node2belong(self, z, u):
        if z not in self.node2belong:
            self.node2belong[z] = dict()
        if u in self.node2belong[z]:
            return self.node2belong[z][u]
        self.node2belong[z][u] = 0
        if z == u:
            self.node2belong[z][u] = 1
            return 1
        fu = self.get_parents(u)
        if z in fu:
            self.node2belong[z][u] = 1 / len(fu)
        else:
            for y in fu:
                self.node2belong[z][u] += self.get_node2belong(z, y) * self.get_node2belong(y, u)
        return self.node2belong[z][u]

    def HOC_cost(self, E):
        root = None
        for u in self.edgeTs:
            if len(self.edgeTs[u]) == 0:
                root = u
        # print("root", root)
        self.get_node2leaf(root)
        logger.debug(self.node2leaf)
        self.init_ancestors()
        cost = 0
        for (u, v, w) in E:
            siz = 0
            mca = self.get_mca(u, v)
            # 隶属度取平凡的1/|fa[u]|
            logger.debug("{} {} {} {}".format(u, v, w, mca))
            for z in mca:
                siz += len(self.node2leaf[z]) / len(mca)
            cost += w * siz
        return cost

    def get_level(self, depth):
        root = None
        for u in self.edgeTs:
            if len(self.edgeTs[u]) == 0:
                root = u
        dep = dict()
        dep[root]=1
        Q = queue.Queue()
        Q.put(root)
        while not Q.empty():
            u = Q.get()
            for v in self.edges[u]:
                if v not in dep:
                    dep[v] = dep[u] + 1
                    Q.put(v)
        S = set()
        for node in self.nodes:
            if dep[node] == depth:
                S.add(node)
        return S

if __name__ == "__main__":
    dag = HOC()
    edges = [
        (0, 'L'),
        (1, 'L'),
        (2, 'L'),
        (3, 'L'),
        (2, 'R'),
        (3, 'R'),
        (4, 'R'),
        ('L', 'root'),
        ('R', 'root'),
    ]
    for (u, v) in edges:
        dag.add_edge(v, u)
    E = [
        (0, 1, 1),
        (0, 2, 1),
        (1, 2, 1),
        (3, 4, 1),
        (2, 3, 1),
    ]
    print(dag.HOC_cost(E))
    print(dag.get_node2belong("L", 2))
