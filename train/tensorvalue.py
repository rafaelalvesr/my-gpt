import numpy as np
import contextlib

def _unbroadcast(grad, shape):
    """
    Sum grad back to the original input shape after broadcasting.
    """
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


def _is_basic_index(key) -> bool:
    if isinstance(key, tuple):
        return all(_is_basic_index(k) for k in key)
    return key is Ellipsis or key is None or isinstance(key, (int, np.integer, slice))


def _normalize_axes(axes, ndim):
    """Normalize possibly negative axes to [0, ndim)."""
    return tuple(ax if ax >= 0 else ax + ndim for ax in axes)


class TensorValue:
    """TensorValue class that supports automatic differentiation for basic operations."""
    no_grad = False #class flag to disable gradient tracking for inference mode

    def __init__(self, data, _children=(), _op="", label="", require_grads=True) -> None:
        self.data = np.asarray(data, dtype=np.float32) # float32 arrays are shared, not copied

        # Determine whether to track gradients based on the global no_grad flag and the instance's require_grads parameter
        track = (not TensorValue.no_grad) and require_grads
        self.grad = np.zeros_like(self.data) if track else None
        self._prev = set(_children) if track else set() #no children since no gradients are tracked
        self.require_grads = require_grads
        self._op = _op
        self._grad_fn = lambda: None # replace with actual gradient function if needed
        self.label = label
        self.shape = self.data.shape

    #Backward function property with setter to allow setting the gradient function while respecting the no_grad flag
    #When no_grad is True, the setter will not set the _grad_fn, effectively disabling gradient tracking for that instance.
    @property
    def _backward(self):
        return self._grad_fn
    
    @_backward.setter
    def _backward(self, fn):
        #if no_grad is False, set the _grad_fn to the provided function; 
        if not TensorValue.no_grad:
            self._grad_fn = fn
        #otherwise, do nothing to disable gradient tracking.


    def __repr__(self):
        return f"TensorValue(data={self.data}, grad={self.grad}, label={self.label})"

    @staticmethod
    def concat(tensors, axis=0):
        """Concatenate a list of TensorValue objects along a specified axis."""
        tensors = [tensor if isinstance(tensor, TensorValue) else TensorValue(tensor) for tensor in tensors]
        if not tensors:
            raise ValueError("concat requires at least one tensor")

        ndim = tensors[0].data.ndim #number of dimensions of the tensors
        axis = axis % ndim #handle negative axis
        out = TensorValue(
            np.concatenate([tensor.data for tensor in tensors], axis=axis),
            tuple(tensors),
            "concat"
        )

        sizes = [tensor.data.shape[axis] for tensor in tensors]
        boundaries = np.cumsum(sizes)[:-1]

        def _backward():
            grad_chunks = np.split(out.grad, boundaries, axis=axis)
            for tensor, grad_chunk in zip(tensors, grad_chunks):
                if tensor.require_grads:
                    tensor.grad += grad_chunk

        out.require_grads = any(tensor.require_grads for tensor in tensors)
        out._backward = _backward #call the setter to set the backward function, which will respect the no_grad flag
        return out
    

    @staticmethod
    def stack(tensors, axis=0):
        tensors = [tensor if isinstance(tensor, TensorValue) else TensorValue(tensor) for tensor in tensors]
        if not tensors:
            raise ValueError("stack requires at least one tensor")

        ndim = tensors[0].data.ndim + 1
        axis = axis % ndim
        out = TensorValue(
            np.stack([tensor.data for tensor in tensors], axis=axis),
            tuple(tensors),
            "stack"
        )

        def _backward():
            for index, tensor in enumerate(tensors):
                grad_slice = np.take(out.grad, index, axis=axis)
                if tensor.require_grads:
                    tensor.grad += grad_slice

        out._backward = _backward
        out.require_grads = any(tensor.require_grads for tensor in tensors)
        return out
    

    def reshape(self, *new_shape):
        out = TensorValue(self.data.reshape(*new_shape), (self,), "reshape")

        def _backward():
            if self.require_grads:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def __add__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other, require_grads=False)

        out = TensorValue(self.data + other.data, (self, other), "+")

        def _backward():
            if self.require_grads:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.require_grads:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        out.require_grads = self.require_grads or other.require_grads
        return out
    
    def add_const(self,other): #other: handle as numpy array constant
        out = TensorValue(self.data + other, (self,), "+const")
        def _backward():
            if self.require_grads:
                self.grad += _unbroadcast(out.grad, self.data.shape)
        out._backward = _backward
        out.require_grads = self.require_grads
        return out
    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other, require_grads=False)
        out = TensorValue(self.data * other.data, (self, other), "*")

        def _backward():
            if self.require_grads:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.require_grads:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        out.require_grads = self.require_grads or other.require_grads
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other, require_grads=False)
        return self * other.pow(-1.0)

    def pow(self, exponent):
        out = TensorValue(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            if self.require_grads:
                self.grad += (exponent * (self.data ** (exponent - 1.0))) * out.grad

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def matmul(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(other, require_grads=False)
        out = TensorValue(self.data @ other.data, (self, other), "@")

        def _backward():
            # For matrix multiplication, the gradients are computed using the chain rule:
            # If out = self @ other, then:
            # grad_self = out.grad @ other.T
            # grad_other = self.T @ out.grad
            grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
            if self.require_grads:
                self.grad += _unbroadcast(grad_self, self.data.shape)
            if other.require_grads:
                other.grad += _unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        out.require_grads = self.require_grads or other.require_grads
        return out
    
    def __rmatmul__(self, other):
        return self.matmul(other)
    
    def T(self):
        """Swap the last two dimensions (matrix transpose semantics)."""
        if self.data.ndim < 2:
            return self
        axes = list(range(self.data.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    def transpose(self, *axes):
        """Transpose tensor dimensions.

        If no axes are given, reverses all dimensions.
        """
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)

        if len(axes) != self.data.ndim:
            raise ValueError(f"transpose expects {self.data.ndim} axes, got {len(axes)}")

        axes = _normalize_axes(axes, self.data.ndim)
        if sorted(axes) != list(range(self.data.ndim)):
            raise ValueError("transpose axes must be a permutation of all dimensions")

        out = TensorValue(np.transpose(self.data, axes), (self,), "transpose")
        inv_axes = tuple(np.argsort(axes))

        def _backward():
            if self.require_grads:
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def permute(self, *axes):
        """Alias for transpose with explicit axis order."""
        return self.transpose(*axes)

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = TensorValue(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                if not keepdims:
                    if isinstance(axis, int):
                        axes = (axis,)
                    else:
                        axes = tuple(axis)
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                grad = np.broadcast_to(grad, self.data.shape)

            if self.require_grads:
                self.grad += grad

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def relu(self):
        out = TensorValue(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            if self.require_grads:
                self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def exp(self):
        exp_data = np.exp(self.data)
        out = TensorValue(exp_data, (self,), "exp")

        def _backward():
            if self.require_grads:
                self.grad += exp_data * out.grad

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def sigmoid(self):
        sig_data = 1.0 / (1.0 + np.exp(-self.data))
        out = TensorValue(sig_data, (self,), "sigmoid")

        def _backward():
            if self.require_grads:
                self.grad += (sig_data * (1.0 - sig_data)) * out.grad

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def log(self):
        safe = np.clip(self.data, 1e-12, None)
        out = TensorValue(np.log(safe), (self,), "log")

        def _backward():
            if self.require_grads:
                self.grad += (1.0 / safe) * out.grad

        out._backward = _backward
        out.require_grads = self.require_grads
        return out

    def backward(self):
        topo,  visited, stack = [], set(), [(self, False)]
        #iterative post-order traversal to build the topological 
        # order of the computational graph
        while stack:
            v, done = stack.pop()
            if done:
                topo.append(v);continue
            if id(v) in visited: continue
            visited.add(id(v))
            stack.append((v, True)) # Post-order traversal: children before parents
            for child in v._prev:
                stack.append((child, False))

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
        
        # teardown the computational graph to free memory after backward pass
        for node in topo:
            node._backward = (lambda: None) #free the closure of the backward function to break references to other nodes
            node._prev = set()

    def zero_grad(self):
        topo,  visited, stack = [], set(), [(self, False)]
        visited = set()

        while stack:
            v, done = stack.pop()
            if done:
                topo.append(v);continue
            if id(v) in visited: continue
            visited.add(id(v))
            stack.append((v, True)) # Post-order traversal: children before parents
            for child in v._prev:
                stack.append((child, False))
        
        for node in topo:
            if node.require_grads:
                node.grad = np.zeros_like(node.data)

    def __getitem__(self, key):
        out = TensorValue(self.data[key], (self,), "getitem")

        def _backward():
            if not self.require_grads:
                return
            if _is_basic_index(key):
                self.grad[key]+=out.grad
            else:
                # Use np.add.at to accumulate gradients for repeated indices
                np.add.at(self.grad, key, out.grad)
        out._backward = _backward
        out.require_grads = self.require_grads
        return out
    

    #Cross-entropy loss (fused log-softmax + negative log-likelihood)
    def cross_entropy(self, targets):
        """
        Fused cross-entropy between logits (self) and integer targets.

            loss = -1/T * Σ_t log_softmax(z)[t, targets[t]]

        Fusing the whole softmax → getitem → log → sum chain into a single
        graph node saves several [T x V]-sized intermediates and lets us use
        the closed-form gradient below.

        Forward — log-sum-exp trick (numerically stable):
            log_softmax(z)_j = (z_j - max(z)) - log( Σ_k exp(z_k - max(z)) )
        The exp argument is always <= 0 (never overflows) and the sum is
        >= 1 (log never sees zero), so no clipping is needed.

        Backward — classic closed-form result:
            L_t = -z_t + log Σ_k exp(z_k)
            ∂L/∂z_j = softmax(z)_j - 1[j == target]   ("softmax minus one-hot")
        The 1/T factor comes from the mean over the T positions.

        Parameters:
            self: [T x V] logits (T positions, V classes)
            targets: [T] integer class indices (e.g. next-token ids)
        Returns:
            scalar TensorValue — mean negative log-likelihood
        """
        z = self.data

        # log_softmax(z)_j = (z_j - max) - log(sum(exp(z_j - max)))
        shifted = z - z.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

        # loss = -1/T * Σ_t log_probs[t, targets[t]]
        seq_len = z.shape[0]
        rows = np.arange(seq_len)
        out = TensorValue(-log_probs[rows, targets].mean(), (self,), "cross_entropy")

        # exp(log_softmax(z)) = softmax(z), kept for the backward pass
        softmax_probs = np.exp(log_probs)

        def _backward():
            # ∂loss/∂z = (softmax(z) - one_hot(targets)) / T, scaled by the
            # upstream gradient (out.grad is a scalar node).
            grad = softmax_probs.copy()  # copy: -= below must not mutate the closure
            grad[rows, targets] -= 1.0   # softmax - one_hot
            if self.require_grads:
                self.grad += (grad / seq_len) * out.grad
        out._backward = _backward
        out.require_grads = self.require_grads
        return out
#End of tensorvalue.py

#Context manager to temporarily set no_grad to True for inference mode, and restore the previous state after exiting the context
@contextlib.contextmanager
def no_grad():
    prev = TensorValue.no_grad
    TensorValue.no_grad = True
    try:
        yield #execute the block of code within the context manager with no_grad set to True
    finally:
        #restore the previous state of no_grad after exiting the context manager
        TensorValue.no_grad = prev
