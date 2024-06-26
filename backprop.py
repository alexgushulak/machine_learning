# https://www.youtube.com/watch?v=VMj-3S1tku0&t=3334s&ab_channel=AndrejKarpathy

import math
import numpy as np
import matplotlib.pyplot as plt
from draw import Visualize

class Value():
    def __init__(self, data: float, _children=(), _op='', label='') -> None:
        self.data: float = data
        self.grad: float = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label: str = label
    
    def __repr__(self) -> str:
        return f"Value(label={self.label}, data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int and float for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad

        out._backward = _backward

        return out
     
    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # topological sorting of the DAG
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron():
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer():
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP():
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



if __name__ == "__main__":
    vz = Visualize()

    # def perceptron():
    #     x1 = Value(2.0, label='x1')
    #     x2 = Value(0.0, label='x2')
    #     w1 = Value(-3.0, label='w1')
    #     w2 = Value(1.0, label='w2')
    #     b = Value(6.8813735870195432, label='b')
    #     x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    #     x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    #     x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1+x2*w2'
    #     n = x1w1x2w2 + b; n.label = 'n'
    #     e = (2*n).exp()
    #     o = (e - 1) / (e + 1); o.label = 'o'
    #     o.backward()
    #     vz.render_graph(o)
    
    x = [2.0, 3.0, -1.0, 5.0]
    n = MLP(2, [2,2,1])
    n(x)
    xs = [
        [2.0, 3.0, -1.0, 4.0],
        [3.0, -1.0, 0.5, 2.0],
        [1.0, 0.0, -1.0, 1.0],
        [1.0, 1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, 1.0]

    learning_rate = 0.05

    for k in range(30):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # backward pass
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -learning_rate * p.grad
        
        print(k, loss.data)

    print("Predictions:")
    print(ypred)

    vz.render_graph(loss)
