import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, input_dim=-1):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.input_dim = input_dim

    def forward(self, x):
        x1,x2 = x.chunk(2, dim=self.input_dim)
        return x1 * self.sigmoid(x2)
    

# Example usage
if __name__ == "__main__":
    glu = GLU()
    torch_glu = torch.nn.GLU()
    x = torch.randn(5, 10)  # Example input
    
    assert torch.allclose(torch_glu(x), glu(x)), "GLU output is not consistent"