from tinygrad import Tensor


class GATLayer:
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 alpha: float = 0.01, dropout: float = 0.6):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.n_heads: int = n_heads
        self.alpha: float = alpha
        self.dropout = dropout

        self.W: Tensor = Tensor.glorot_uniform(
            in_features, out_features * n_heads)
        self.a: Tensor = Tensor.glorot_uniform(n_heads, 2 * out_features, 1)

    def __call__(self, x: Tensor, adjacency_matrix: Tensor) -> Tensor:
        n_nodes = x.shape[0]

        x = x.dropout(self.dropout)
        x = x.linear(self.W)
        x = x.dropout(self.dropout)
        x = x.view(n_nodes, self.n_heads, self.out_features).permute(1, 0, 2)

        source_scores = Tensor.einsum(
            'bij,bjk->bik', x, self.a[:, :self.out_features, :])
        target_scores = Tensor.einsum(
            'bij,bjk->bik', x, self.a[:, self.out_features:, :])

        attention = source_scores + target_scores.transpose(-2, -1)
        attention = attention.leakyrelu(self.alpha)

        connectivity_mask = Tensor.full(attention.shape, float('-inf'))
        attention = Tensor.where(adjacency_matrix > 0,
                                 attention, connectivity_mask)

        attention = attention.softmax(-1)
        attention = attention.dropout(self.dropout)

        x = Tensor.einsum('bij,bjk->bik', attention, x)
        x = x.mean(0)

        return x
