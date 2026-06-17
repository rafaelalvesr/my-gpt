import math
from typing import Tuple

class Value:
    def __init__(self, data: float, _prev: Tuple['Value', ...] = (), _op: str = '', label: str = '') -> None:
        """
        children: the data objects that were used to compute this data
        op: the operation that was used to compute this data (e.g., '+', '*', etc.)
        """
        self.data = data
        # self.grad = d out/ d self
        #grad is the gradient of the output (final data of the computation) with respect to this data,
        #  which will be computed during backpropagation
        self.grad = 0 
        self._prev = _prev
        self._op = _op
        self._backward = lambda: None
        self.label = label

    def __add__(self, other):
        #convert other to a data if it is not already a data (e.g., if other is a number)
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        #out = self.data + other.data (local output)
        out = Value(self.data + other.data, _prev=(self, other), _op='+', label='+')
        #f = final output of the computation
        #d out/ d self = 1 and  d out/ d other = 1
        # d f/ d self = d f/ d out * d out/ d self = d f/ d out * 1 = d f/ d out
        # d f/ d other = d f/ d out * d out/ d other = d f/ d out * 1 = d f/ d out
        def _backward():
            #we add to the gradients of self and other because they might be used in multiple places in the computation graph, 
            # so we need to accumulate the gradients from all the paths that lead to self and other
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        #if a.__add__(b) fail, then it will try b.__radd__(a), which will call this method.
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        # 5 - x -> 5 + (-x)
        return other + (-self)
    
    def __truediv__(self, other):
        #a/b -> a*(1/b) -> a*(b**-1)
        return self * other**-1
    
    def __rtruediv__(self, other):
        # 5 / x -> 5 * (x**-1)
        return Value(other) / self
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        #out = self.data * other.data (local output)
        out = Value(self.data * other.data, _prev=(self, other), _op='*', label='*')
        #f = final output of the computation graph
        # d out/ d self = other.data and  d out/ d other = self.data
        # d f/ d self = d f/ d out * d out/ d self = d f/ d out * other.data
        # d f/ d other = d f/ d out * d out/ d other = d f/ d out * self.data
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        #out = self.data ** other (local output)
        out = Value(self.data ** other, _prev=(self,), _op=f'**{other}', label='**')
        #f = final output of the computation graph
        # d out/ d self = other * self.data ** (other - 1)
        # d f/ d self = d f/ d out * d out/ d self = d f/ d out * other * self.data ** (other - 1)
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        #out = exp(self.data) (local output)
        x = self.data
        _exp = math.exp(x)
        out = Value(_exp, _prev=(self,), _op='exp', label='exp')
        #f = final output of the computation graph
        # d out/ d self = exp(self.data)
        # d f/ d self = d f/ d out * d out/ d self = d f/ d out * exp(self.data)
        def _backward():
            self.grad += _exp * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        #out = tanh(self.data) (local output)
        x = self.data
        _tanh = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(_tanh, _prev=(self,), _op='tanh', label='tanh')
        #f = final output of the computation graph
        # d out/ d self = 1 - tanh(self.data)**2
        # d f/ d self = d f/ d out * d out/ d self = d f/ d out * (1 - tanh(self.data)**2)
        def _backward():
            self.grad += (1 - _tanh**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        #out = relu(x) = max(0, x) (local output)
        _relu = x if x > 0 else 0
        out = Value(_relu,_prev = (self,), _op = 'relu', label='relu')
        #f = final output of the computation graph
        # d out/ d self = 1 if x > 0 else 0
        # d f/ d self = d f/ d out * d out/ d self = d f/ d out * (1 if x > 0 else 0)
        def _backward():
            self.grad += (1 if x > 0 else 0) * out.grad
        out._backward = _backward
        return out
    
    #Natural logarithm
    def log(self):
        x = max(self.data, 1e-10)
        #out = log(x) (local output)
        _log = math.log(x)
        out = Value(_log, _prev=(self,), _op='log', label='log')
        #f = final output of the computation graph
        # d out/ d self = 1 / x
        # d f/ d self = d f/ d out * d out/ d self = d f/ d out * (1 / x)
        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        # reset the gradients of all the variables in the graph to zero
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for v in topo:
            v.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, label={self.label})"
    
    def __str__(self):
        return f"{self.data}"
    
    def print_children(self):
        print(f"Children of {self}:")
        for child in self._prev:
            print(f"  {child} (op: {child._op}, grad: {child.grad} data: {child.data})")