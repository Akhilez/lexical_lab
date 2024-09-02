import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(embedding_size))
        self.gamma = nn.Parameter(torch.ones(embedding_size))

    def forward(self, x):
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
        return self.gamma * x / (rms + 1e-8) + self.alpha


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, n_heads):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads

        assert embedding_size % n_heads == 0, "Embedding size must be divisible by number of heads"

        self.qkv = nn.Linear(embedding_size, 3 * embedding_size)
        self.project = nn.Linear(embedding_size, embedding_size)

        self.norm1 = RMSNorm(embedding_size)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Linear(embedding_size * 4, embedding_size),
        )

        self.norm2 = RMSNorm(embedding_size)

    def forward(self, x):
        b, s, e = x.size()
        h = self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # (b, s, 3 * e) -> (b, s, e) x 3

        # (b, s, e) -> (b, s, h, e) -> (b, h, s, e)
        q = q.view(b, s, h, e // h).transpose(1, 2)
        k = k.view(b, s, h, e // h).transpose(1, 2)
        v = v.view(b, s, h, e // h).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (b, h, s, e)  # FlashAttention
        y = y.transpose(1, 2).contiguous().view(b, s, e)  # re-assemble all head outputs side by side
        y = self.project(y)  # (b, s, e)

        y = self.norm1(y + x)  # Add and norm

        x = self.feedforward(y)
        y = self.norm2(x + y)  # Add and norm

        return y


class AutoRegressiveLM(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_seq_length, n_layers, n_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_length = max_seq_length
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional = nn.Embedding(max_seq_length, embedding_size)
        self.attn_blocks = nn.ModuleList([AttentionBlock(embedding_size, n_heads) for _ in range(n_layers)])
        self.emb2vocab = nn.Linear(embedding_size, vocab_size)

        # Tie weights
        self.emb2vocab.weight = self.embedding.weight

    def forward(self, x):
        b, s = x.shape
        assert s <= self.max_seq_length, "Sequence length exceeds max_seq_length"
        x = self.embedding(x)
        x = x + self.positional(torch.arange(s, device=x.device))[None, :, :]
        for attn_block in self.attn_blocks:
            x = attn_block(x)
        x = self.emb2vocab(x)
        return x

    def generate(self, tokens, length, top_k=50, temperature=1.0):
        tokens = tokens.detach().clone()  # (b, s)
        for _ in range(length):
            logits = self.forward(tokens)  # (b, s, v)
            logits = logits[:, -1, :]  # (b, v)  # only take the last token
            probs = F.softmax(logits / temperature, dim=-1)  # (b, v)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)  # (b, k)
            next_token_indices = torch.multinomial(topk_probs, num_samples=1)  # (b, 1)
            next_token = torch.gather(topk_indices, -1, next_token_indices)  # (b, 1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            tokens = tokens[:, -self.max_seq_length:]  # only keep the last max_seq_length tokens
        return tokens


if __name__ == '__main__':
    vocab_size = 64
    embedding_size = 32
    max_seq_length = 128
    n_layers = 4
    n_heads = 4  # 32 / 4 = 8
    batch_size = 2

    model = AutoRegressiveLM(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        max_seq_length=max_seq_length,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    print(model)

    x = torch.randint(0, vocab_size, (batch_size, max_seq_length), dtype=torch.long)
    y = model(x)
    print(y.shape)

    from torchsummary import summary
    summary(model, x, depth=10)

    tokens = torch.randint(0, vocab_size, (1, 10), dtype=torch.long).repeat(batch_size, 1)
    print(tokens)
    generated = model.generate(tokens, length=max_seq_length - 10 + 1, top_k=50, temperature=1.0)
    print(generated.shape)
    print(generated)
