## What is a Transformer?
- Fundamentally, text-generative Transformer models operate on the principle of next-token prediction: given a text prompt from the user, what is the most probable next token (a word or part of a word) that will follow this input? 
- The core innovation and power of Transformers lie in their use of self-attention mechanism, which allows them to process entire sequences and capture long-range dependencies more effectively than previous architectures.
![Transformer](images/Transformer.png)

## Transformer Architecture Components
- Embedding
- Transformer Block
- Output Probabilities

### 1. Embedding
#### Process of transforming text into a numerical representation that the model can work with
** The output of this Embedding process is a Sentence matrix(also called input embeddings). The shape of this is (#INPUT_TOKENS, EMBEDDING_DIM)**

Process/Steps Involved:
1. Tokenization 
2. Token Embedding
3. Positional Encoding
4. Final Embedding

- Tokenization is the process of breaking down the input text into smaller, more manageable pieces called tokens. 

**The full vocabulary of tokens is decided before training the model: GPT-2's vocabulary has 50,257 unique tokens.**

- Each token in the vocabulary is represented as a 768-dimensional vector.GPT-2's entire vocabulary is stored in an Token Embedding Matrix(an embedding lookup table) of shape (50257, 768). Shape of Token Embedding Matrix is (VOCAB_SIZE, EMBEDDING_DIM).

**How it works:** When the model sees token ID 154, it goes to row 154 of this matrix and copies those numbers.

![Embedding Process](images/embedding.png)

| Model | Architecture | Vocab Size | Embedding Dim ($d_{model}$) | Tokenizer Type | Unique Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT** (Base) | Encoder-only | 30,522 | 768 | WordPiece | Introduced Masked Language Modeling (MLM); separate `[CLS]` and `[SEP]` tokens. |
| **GPT-2** (Small) | Decoder-only | 50,257 | 768 | Byte-level BPE | Introduced the idea that "Language Models are Unsupervised Multitask Learners"; standard absolute positional embeddings. |
| **RoBERTa** | Encoder-only | 50,265 | 768 (Base)<br>1024 (Large) | Byte-level BPE | A robustly optimized BERT; removed Next Sentence Prediction (NSP); increased vocab size to match GPT-2 style. |
| **T5** (Base) | Encoder-Decoder | 32,128 | 768 | SentencePiece | Unified "Text-to-Text" framework; uses **Relative Positional Encodings** rather than absolute. |
| **GPT-3** (175B) | Decoder-only | 50,257 | 12,288 | Byte-level BPE | Massive scale up of GPT-2 architecture; uses Alternating Dense and Locally Banded Sparse Attention. |
| **LLaMA 1 & 2** (7B) | Decoder-only | 32,000 | 4,096 | SentencePiece (BPE) | Popularized **RoPE** (Rotary Positional Embeddings) and **SwiGLU** activation functions; Pre-normalization (RMSNorm). |
| **Mistral 7B** | Decoder-only | 32,000 | 4,096 | SentencePiece (BPE) | Uses **Sliding Window Attention** (SWA) and **Grouped-Query Attention** (GQA) for faster inference. |
| **LLaMA 3** (8B) | Decoder-only | **128,256** | 4,096 | Tiktoken (BPE) | Significant jump in vocab size compared to LLaMA 2; allows for much better data compression (fewer tokens per sentence). |
| **Gemma** (7B) | Decoder-only | **256,000** | 3,072 | SentencePiece (Proto) | Massive vocabulary size; GeGLU activations; normalizes input embeddings by $\sqrt{d_{model}}$. |
| **GPT-4** | Decoder-only (MoE) | ~100,277* | Unknown | Tiktoken (`cl100k`) | *Estimated based on OpenAI's `cl100k_base` tokenizer.* Likely a Mixture of Experts (MoE) architecture. |

---

**Old Models used vocab size of 30-50k while modern,newer models use 128k+ vocab size.**
- **Larger Vocab = Higher Information Density. A single token can represent a whole complex word. This effectively increases the context window because the same sentence requires fewer tokens.(i.e Larger Vocab = fewer tokens per sentence)**
- **Tradeoff is large memory and final output layer (Logits) becomes massive, requiring significant compute to calculate probabilities over 128k or 256k possibilities.(transformer predicts the next token based on current input tokens, so basically like too many classes present to choose the next token from)**

- In GPT-2 (and many other transformers), the Token Embedding Matrix is often "tied" to the output layer. This means the matrix used to turn inputs into vectors is the exact same matrix used at the very end to turn the final hidden states back into probabilities for the next word. This is known as **Weight Tying**.

Related Research: 
1. [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies](https://arxiv.org/abs/2407.13623)
2. [Large Vocabulary Size Improves Large Language Models](https://aclanthology.org/2025.findings-acl.57.pdf)

### 2. Transformer Block
#### Transformer Block consists of multi-head self-attention and a Multi-Layer Perceptron layer. 
- Most models use multiple such blocks that are stacked sequentially one after the other and the token representations evolve through layers, from the first block to the last one
Eg: The GPT-2 (small) model consists of 12 such blocks.

#### A) Multi-Head Self-Attention (MHSA)
- The **Self-Attention mechanism** enables the model to capture relationships among tokens in a sequence, so that each token’s representation is influenced by the others.
- **Multiple attention heads** allow the model to consider these relationships from **different perspectives**; for example, one head may capture short-range syntactic links while another tracks broader semantic context.

Process/Steps Involved:
1. Query, Key, and Value Matrices
2. Multi-Head Splitting
3. Masked Self-Attention
4. Output and Concatenation

##### 1. Query, Key, and Value Matrices
* Each token embedding vector is transformed into 3 dimensions(3 basis vectors): Query (Q), Key (K), and Value (V). These vectors are derived by multiplying the input embedding matrix with learned weight matrices for Q, K, and V.
![Transforming Embeddings into QKV](images/QKV1.png)
**The choice of 3 vectors is based on Information Retrieval Theory and the necessity of Asymmetric Relationships in language.**

---

##### A) Language is Assymetric:
If we did not transform the input X and instead just used the input vector X for everything, the attention score would be: X.X<sup>T</sup>

For a phrase "river bank", bank is dependent on river to determine if it is related to nature or is a financial institution but river is not dependent on bank. So, when the token for bank is to be predicted by the model, it needs context from "river" token and not viceversa. This is the Asymmetric Relationship in Languages. Using a single dimension defeats this purpose as dot product of 
matrices/vectors is symmetric. 

##### B) Information Retrieval Theory:
Imagine you are looking for a book in a library database. The process involves three distinct components:

- Query (Q): What you type into the search bar (e.g., "books about space").
- Key (K): The metadata/labels on the books in the database (e.g., "Astronomy", "Sci-Fi", "Cosmology").
- Value (V): The actual content of the book you pull off the shelf.

We need atleast 3 dimensions(including the input) to retrieve the information we need.

---
##### Why not higher dimensions?
We can have more than 3 dimensions for retrieving the info(in transformers case, it is relevant tokens) but it is redundant as having more dimensions is not enriching the information(i.e rank of the matrix is not increasing)

Explanation:

If your input embedding x has a dimension of 768, it contains a finite amount of information. Geometrically, the data points sit in a 768-dimensional subspace.
If you project this 768 vector into a 10,000-dimensional space using a matrix multiplication (768×10000), the resulting vector is technically 10,000 numbers long, but it still only has a mathematical rank of 768.

**Rank of Matrix:** The rank of a matrix is the maximum number of linearly independent row or column vectors, representing the dimension of its vector space.
i.e How many dimensions/numbers in the matrix actually matter or contribute to the information it represents.

K,Q,V Matrices:
If we have a vector denoting force of say 5N, we can transform it into x,y,z components as these are the basis vectors for the Cartesian coordinate system. Similarly for an embedding vector with 768 dimension, we need to multiply it with 768 scalars to project them into K,Q,V dimensions. But since these dimensions are not standard/fixed like our Cartesian coordinate system, we learn them using 3 matrices instead.

##### 2. Multi-Head Splitting
Once Q,K,V matrices are computed(after multiplying input embedding matrix with Q,K,V weight matrices), they are split into multiple heads by reshaping the embedding dimension into (num_heads, head_dim).

i.e (seq_len, embed_dim) -> (seq_len, num_heads, head_dim) or (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)

In case of GPT-2 model, embed_dim is 768 and num_heads is 12. So, head_dim is 768/12 = 64.
(6,768) -> (6,12,64)

![Multihead-Splitting](images/Multihead-Splitting.png)

**Once each of Q,K,V vectors are transformed, we perform a transform to bring the head dimension to the first dimension. i.e (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim)**

![Multihead-Splitting-2](images/Multihead-Splitting-2.png)

##### 3. Multi Head Attention
- Multi-Head Self-Attention (MHSA) is the core operation in the Transformer architecture.
- Transformer uses multiple heads to capture different relationships among tokens in a sequence.
- The part that multiplies Q projection with K<sup>T</sup> and scale it down using the square root of the dimension of the key vectors is called **Self-Attention**.(Math formula is Scaled Dot-Product Attention, SDPA)
- In transformer encoder part, each head computes this Self-Attention and while in decoder part, it computes this Self-Attention with a mask, thus it is called **Masked Self-Attention**.
![Masked-Self-Attention](images/Masked-Self-Attention.png)

![MHABlock](images/MHABlock.png)


## MHSA Shape Transformations
> Reference: GPT-2 Small — `seq_len=6`, `embed_dim=768`, `num_heads=12`, `head_dim=64`, `vocab_size=50257`

---

## Pre-Block (runs once)

| Step | Operation | Input Shape | Output Shape | Description |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Token + Positional Embedding | `(6,)` | `(6, 768)` | Each token ID is looked up in the embedding table and summed with its positional encoding |

---

## Transformer Block × N (repeats 12× in GPT-2 parallelly)

### A) Multi-Head Self-Attention (MHSA)

| Step | Operation | Input Shape | Output Shape | Description |
| :---: | :--- | :---: | :---: | :--- |
| 2 | QKV Projection | `(6, 768)` | `(6, 2304)` | Input matrix is multiplied by the QKV weight matrices W_Q, W_K, W_V [each of shape **(embed_dim, embed_dim)** ] to produce all three projections at once |
| 3 | Split Q, K, V | `(6, 2304)` | `3 × (6, 768)` | The 2304-dim output is sliced into three separate Q, K, V matrices each of 768 dims |
| 4 | Reshape into heads | `(6, 768)` | `(6, 12, 64)` | Each of Q, K, V is reshaped — the 768 embedding dim is split into 12 heads of 64 dims each |
| 5 | Transpose | `(6, 12, 64)` | `(12, 6, 64)` | Head dimension is moved to front so each head operates independently on all 6 tokens |
| 6 | Scores — Q × Kᵀ | `(12, 6, 64) × (12, 64, 6)` | `(12, 6, 6)` | Every token's query is dot-producted against every token's key to produce one relevance score per pair |
| 7 | Scale | `(12, 6, 6)` | `(12, 6, 6)` | Each score is divided by √64 = 8 to **prevent vanishing gradients** from large dot products |
| 8 | Mask *(decoder only)* | `(12, 6, 6)` | `(12, 6, 6)` | Future token positions are filled with −∞ so they become zero after softmax **(add a matrix of same shape with upper triangle as -inf)** |
| 9 | Softmax | `(12, 6, 6)` | `(12, 6, 6)` | Each row of scores is converted to a probability distribution summing to 1 |
| 10 | Weighted sum — AttnWeights × V | `(12, 6, 6) × (12, 6, 64)` | `(12, 6, 64)` | Each token's output is a weighted blend of all Value vectors according to the attention weights |
| 11 | Transpose back | `(12, 6, 64)` | `(6, 12, 64)` | Head dimension is moved back after position so heads can be concatenated along the embedding axis |
| 12 | Reshape — concatenate heads | `(6, 12, 64)` | `(6, 768)` | All 12 heads are concatenated back into a single 768-dim vector per token |
| 13 | Output projection W_O | `(6, 768)` | `(6, 768)` | Learned linear layer re-mixes information across all heads into the final attention output |
| 14 | Residual + LayerNorm | `(6, 768)` | `(6, 768)` | Original input is added back to the attention output and normalised for training stability |

> **Steps 4–5 and 11–12** are free memory reshapes — no computation, no learned parameters.
> **Steps 6–13** are the core MHSA computation.

---

### B) Feed-Forward Network (FFN)

| Step | Operation | Input Shape | Output Shape | Description |
| :---: | :--- | :---: | :---: | :--- |
| 15 | FFN up-projection (W_1) | `(6, 768)` | `(6, 3072)` | Each token is independently projected up to 4× the embedding dim |
| 16 | Activation (GELU) | `(6, 3072)` | `(6, 3072)` | Non-linearity applied element-wise, enabling the network to learn non-linear relationships |
| 17 | FFN down-projection (W_2) | `(6, 3072)` | `(6, 768)` | Each token is projected back down to embedding dim |
| 18 | Residual + LayerNorm | `(6, 768)` | `(6, 768)` | FFN input is added back and normalised — completes one full Transformer block |

---

## Post-Block (runs once)

| Step | Operation | Input Shape | Output Shape | Description |
| :---: | :--- | :---: | :---: | :--- |
| 19 | Final linear (W_vocab) | `(6, 768)` | `(6, 50257)` | Last token's hidden state is projected to vocabulary size to produce raw logit scores |
| 20 | Softmax | `(6, 50257)` | `(6, 50257)` | Logits are converted to probabilities — the model samples the next token from this distribution |

---

## Summary

| Phase | Steps | Repeats |
| :--- | :---: | :--- |
| Embedding | 1 | Once |
| Transformer Block (MHSA + FFN) | 2–18 | 12× in GPT-2 |
| Output Projection | 19–20 | Once |

Sources:
- https://krypticmouse.hashnode.dev/attention-is-all-you-need
- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse
- https://poloclub.github.io/transformer-explainer/
- https://www.projectpro.io/article/multi-head-attention-in-transformers/1166
- https://www.vizuaranewsletter.com/p/why-do-we-need-masking-in-attention