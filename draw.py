from graphviz import Digraph


class Visualize():
    def __init__(self):
        pass

    def trace(self, root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    def draw_dot(self, root):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # left  to right
        nodes, edges = self.trace(root)

        for n in nodes:
            uid = str(id(n))
            dot.node(name = uid, label = " { %s | data: %.4f | grad: %.4f }" % (n.label, n.data, n.grad), shape='record')
            if n._op:
                dot.node(name = uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        return dot
    
    def render_graph(self, root):
        self.draw_dot(root).render('visualization', format='png', cleanup=True)