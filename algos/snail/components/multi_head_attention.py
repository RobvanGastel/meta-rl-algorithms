from torch import nn

from algos.snail.components.scaled_dot_product import ScaledDotProduct


class MultiHeadAttention(nn.Module):
    def __init__(self, model_d, n_head):
        super().__init__()
        # One crucial characteristic of the multi-head attention is that it
        # is permutation-equivariant with respect to its inputs.

        self.n_head = n_head

        self.attention = ScaledDotProduct()
        self.W_v = nn.Linear(model_d, model_d)
        self.W_k = nn.Linear(model_d, model_d)
        self.W_q = nn.Linear(model_d, model_d)
        self.W_concat = nn.Linear(model_d, model_d)

    def forward(self, Q, K, V, mask=None, return_attention=False):

        # 1) Weight matrix projections
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)

        # 2) Split attention by number of heads
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        # 3) Scaled dot product for similarity
        values, attention = self.attention(Q, K, V, mask=mask)

        # 4) Concat the heads and apply projection
        values = self.concat(values)
        values = self.W_concat(values)

        # TODO: Attention map visualization?

        if return_attention:
            return values, attention
        else:
            return values

    def concat(self, tensor):
        # Concat layers after heads
        # tensor shape: [batch, head, length, d_split]
        # return shape: [batch, timestep, model_d]

        batch_size, head, length, d_split = tensor.size()
        model_d = head * d_split

        return tensor.transpose(1, 2).contiguous().view(batch_size, length, model_d)

    def split(self, tensor):
        # Split tensor by number of head
        #
        # tensor shape: [batch, timestep, model_d]
        # return shape: [batch, head, length, d_split]

        batch_size, length, model_d = tensor.size()

        d_split = model_d // self.n_head
        return tensor.view(batch_size, length, self.n_head, d_split).transpose(1, 2)
