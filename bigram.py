import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
import tiktoken

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
dropout = 0.2
num_heads = 6
n_layer = 6
DATA = "shakespeare"  # potter, shakespeare
SUBWORD_TOKENIZER = True
PATH = f'/home2/manugaur/myGPT/{DATA}_model.pt'
if SUBWORD_TOKENIZER:
    PATH = f'/home2/manugaur/myGPT/{DATA}_tiktoken_model.pt'
TRAIN = False
# -----------------------------------------------------------------------------------------

torch.manual_seed(1337)

# wget  /myGPT/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(f'/home2/manugaur/myGPT/{DATA}.txt', 'r', encoding='utf-8') as f:
    text = f.read()


## CREATE TOKENIZER
# here are all the unique characters that occur in this text
if SUBWORD_TOKENIZER:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    encoded_text = encode(text)
    # vocab_size = len(set(encoded_text))
    vocab_size = 50304
    print(f'Vocab size = {vocab_size}')
    data = torch.tensor(encoded_text, dtype=torch.long)

else:
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    print(f'Vocab size = {vocab_size}')
    data = torch.tensor(encode(text), dtype=torch.long)

# Train and test splits
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# SAMPLING A BATCH OF BLOCKS FROM A TEXT SPLIT

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,)) #sample random starting indices from the data for {batch_size} blocks to be sampled.
    x = torch.stack([data[i:i+block_size] for i in ix]) #sample those blocks, stack them as rows.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # sample ALL {block_size} the targets for a block, stack them.
    return x.to(device), y.to(device)
## VISUALIZING BATCHES
# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")
#     print('\n')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



# single head of self attention 
class Head(nn.Module):
  
  def __init__(self, proj_dim):
    super().__init__()

    self.query = nn.Linear(n_embed, proj_dim)
    self.key = nn.Linear(n_embed, proj_dim)
    self.value = nn.Linear(n_embed, proj_dim)
    self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout) 

  def forward(self, x):
    B,T,C = x.shape
    q = self.query(x)
    k = self.key(x)
    ## -2,-1 vs 1,2  
    w = q@k.transpose(-2,-1)*C**(-0.5)
    w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
    w =  F.softmax(w, dim = -1) # (B,T,T)
    w = self.dropout(w) #sparsity in token interaction. limit communication amongst some tokens.
    v = self.value(x)

    ## creating an output for each query

    out = w @ v
    return out

class MultiHeadAttention(nn.Module):
    # works in parallel
    # instead of taking converting tokens from N-dim to N-dim but with context.
    # we convert N-dim vectors to N/num_head dim vectors. 
    # which are then concatenated to give a N-dim vector.

    def __init__(self, num_heads, proj_dim):
        super().__init__( )
        self.heads = nn.ModuleList([Head(proj_dim) for _ in range(num_heads)])
        self.proj =  nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        attn_out = torch.cat([h(x)for h in self.heads], dim = -1)
        return self.dropout(self.proj(attn_out))


class FeedForward(nn.Module):
    # all the tokens use attention to communicate with each other.
    # attn updates the representations of each token
    #ffw computation is done at the token level.
    # each token has a ffw. To process the context they gained from attention.

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
            # nn.Linear(n_embed, n_embed)
            nn.Dropout(dropout),
        )
    def forward(self, x ):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed//num_heads
        self.attn = MultiHeadAttention(num_heads, head_size)
        self.ffw = FeedForward(n_embed)

        """
        Layernorm is applied to each token respectively.BxT becomes the batch.
        It ensures that the mean and var of token embeddings across time dimension = 0, 1 resp.
        BatchNorm does the same but across batch dimension.
        
        """
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embed_table = nn.Embedding(block_size, n_embed)
        # self.mha = MultiHeadAttention(num_heads,int(n_embed/num_heads))
        # self.attn_head = Head(n_embed)
        # self.ffw = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # for a batch b and timestep t, we get a token embedding. Its a pdf which use to sample the next token.
        token_embed= self.token_embedding_table(idx) # (B,T,n_embed)

        # for each timestep, we get a corresponding positional embedding.
        pos_embed = self.pos_embed_table(torch.arange(T, device = device)) # (B, T, n_embed)

        # For all batches, token embedding of each timestep, has its corresponding P.E added to it.
        x = token_embed + pos_embed  # (B, T, n_embed)
        # attention aggregation 
        # x = self.attn_head(x)
        # x = self.mha(x) #(B, T, n_embed)
        # ffw
        # x = self.ffw(x) # (B, T, n_embed)       

        """
        Blocks of communication followed by computation
        """
        x = self.blocks(x)
        #logits are vocab_size. can sample next word from its softmax distribution.
        x = self.final_ln(x) 
        logits = self.lm_head(x)  # (B,T,vocab_size)
    

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # import pdb;pdb.set_trace()

        for _ in range(max_new_tokens):
            # get the predictions
            idx_block = idx[:, -block_size:]
            logits, loss = self(idx_block)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            print(decode(idx_next[0].tolist()), end = '')

            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

#create a PyTorch optimizer
if TRAIN:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    min_val_loss = float(1000)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val']<min_val_loss:
                min_val_loss =losses['val'].item()
                torch.save(m.state_dict(), PATH)

            
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
else:
    # generate from the model
    m.load_state_dict(torch.load(PATH))
    m.eval()
    x = "Thou leave me "
    context = torch.tensor(encode(x), dtype = torch.long, device = device).unsqueeze(0)

    # _ = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(x, end = '')
    m.generate(context, max_new_tokens= 1000)[0].tolist()
    print('')
