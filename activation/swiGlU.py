import torch
import torch.nn as nn

class SwiGlU(nn.Module):
    def __init__(self):
        super(SwiGlU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        return x * self.sigmoid(x)
    


# Example usage
if __name__ == "__main__":
    swi_glu = SwiGlU()
    torch_swi_glu = torch.nn.SiLU()
    x = torch.randn(5, 10)  # Example input

    assert torch.allclose(torch_swi_glu(x), swi_glu(x)), "SwiGLU output is not consistent"