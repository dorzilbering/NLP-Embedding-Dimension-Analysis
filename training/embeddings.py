"""Internal lookup bottleneck initialized from pretrained token weights, not text."""
import torch
from torch import nn


class TrainableTokenEmbedding(nn.Module):
    """V x d lookup followed by an optional learned d -> H input projection.

    Reduced initialization is uncentered truncated SVD via a streamed W.T @ W
    eigendecomposition in CPU float64. With orthonormal leading vectors Q,
    lookup = W @ Q and projection.weight = Q. Thus effective token weights are
    W @ Q @ Q.T. No sentence vectors, dataset examples, or PCA are involved.
    """

    def __init__(self, vocab_size, dimension, hidden_size, padding_idx=None):
        super().__init__()
        if type(dimension) is not int or not 0 < dimension <= hidden_size:
            raise ValueError("Token dimension must be positive and <= hidden size.")
        self.num_embeddings = vocab_size
        self.embedding_dim = dimension
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.output_dtype = torch.float32
        self.lookup = nn.Embedding(vocab_size, dimension, padding_idx=padding_idx)
        self.projection = (nn.Identity() if dimension == hidden_size
                           else nn.Linear(dimension, hidden_size, bias=False))
        self.initialization = {}

    @classmethod
    def from_pretrained(cls, original, dimension, chunk_size=4096):
        if not isinstance(original, nn.Embedding):
            raise ValueError("Expected a plain pretrained nn.Embedding.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        weight = original.weight.detach()
        vocab, hidden = weight.shape
        # All temporary random weights are overwritten. Do not let their size
        # alter the later classifier initialization for the same run seed.
        with torch.random.fork_rng(devices=[]):
            module = cls(vocab, dimension, hidden, original.padding_idx)
        module.output_dtype = weight.dtype
        with torch.no_grad():
            if dimension == hidden:
                module.lookup.weight.copy_(weight.cpu().float())
                module.initialization = {"method": "pretrained_copy", "relative_error": 0.0}
            else:
                gram = torch.zeros(hidden, hidden, dtype=torch.float64)
                for start in range(0, vocab, chunk_size):
                    block = weight[start:start + chunk_size].to(device="cpu", dtype=torch.float64)
                    gram.add_(block.T @ block)
                values, vectors = torch.linalg.eigh(gram)
                basis = vectors[:, -dimension:].flip(1).contiguous()
                # Fix eigenvector sign ambiguity. Degenerate eigenspaces can still
                # differ across BLAS/platforms; persist the actual initialized weights.
                pivots = basis.abs().argmax(dim=0)
                signs = basis[pivots, torch.arange(dimension)].sign()
                basis.mul_(signs)
                for start in range(0, vocab, chunk_size):
                    block = weight[start:start + chunk_size].to(device="cpu", dtype=torch.float64)
                    module.lookup.weight[start:start + chunk_size].copy_(block @ basis)
                module.projection.weight.copy_(basis)
                energy = values.clamp_min(0)
                total = energy.sum()
                if not torch.isfinite(total) or total <= 0:
                    raise ValueError("Invalid pretrained embedding matrix energy.")
                error = ((energy[:-dimension].sum() / total).sqrt()).item()
                module.initialization = {"method": "uncentered_token_weight_truncated_svd",
                                         "relative_error": error, "gram_dtype": "float64",
                                         "chunk_size": chunk_size}
        return module

    def forward(self, input_ids):
        # Trainable tables/projections remain float32, even with a BF16 backbone.
        return self.projection(self.lookup(input_ids)).to(dtype=self.output_dtype)
