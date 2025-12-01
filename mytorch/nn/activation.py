import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        self.A = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True)) / np.sum(np.exp(Z - np.max(Z, axis=self.dim, keepdims=True)), axis=self.dim, keepdims=True)
        # Apply it to the dimension specified by the `dim` parameter
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = np.moveaxis(self.A, self.dim, -1)
            self.A = np.reshape(self.A, (-1, self.A.shape[-1]))
            dLdA = np.moveaxis(dLdA, self.dim, -1)
            dLdA = np.reshape(dLdA, (-1, dLdA.shape[-1]))
            
        B, _ = self.A.shape
        dLdZ = np.zeros((B, C))
        for i in range(B):
            J = np.zeros((C, C))
            for m in range(C):
                for n in range(C):
                    J[m][n] = self.A[i][m] * (1 - self.A[i][m]) if m == n else (-1) * (self.A[i][m]) * (self.A[i][n])

            row = dLdA[i] @ J
            dLdZ[i, :] = row

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(*shape[:-1], C)
            self.A = np.moveaxis(self.A, -1, self.dim)
            dLdZ = dLdZ.reshape(*shape[:-1], C)
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)

        return dLdZ
 

    