# Conditional Random Fields

This package currently provides support for Linear Chain Conditional Random Fields.

Let us first load the dependencies:

```@example crf
using Flux
using Flux: onehot, LSTM, Dense, reset!
using TextModels: CRF, viterbi_decode, crf_loss
nothing # hide
```

Conditional Random Field layer is essentially like a softmax layer that operates on the top-most layer.

Let us suppose the following input sequence to the CRF with `NUM_LABELS = 2`:

```@example crf
using Random
Random.seed!(42) # For reproducible documentation
NUM_LABELS = 2
SEQUENCE_LENGTH = 3 # CRFs can handle variable length inputs sequences
input_seq = [rand(NUM_LABELS + 2) for i in 1:SEQUENCE_LENGTH] # NUM_LABELS + 2, where two extra features correspond to the :START and :END label.
```

We define our CRF layer as:

    CRF(NUM_LABELS::Integer)

```@example crf
c = CRF(NUM_LABELS) # The API internally append the START and END tags to NUM_LABELS.
```

Now for the initial variable in Viterbi Decode or Forward Algorithm,
we define our input as:

```@example crf
init_α = fill(-10000, (c.n + 2, 1))
init_α[c.n + 1] = 0
init_α
```

Optionally this could be shifted to GPU by `init_α = gpu(init_α)`,
considering the input sequence to be a CuArray in this case.
To shift a CRF `c` to GPU, one can use `c = gpu(c)`.

To find the CRF loss, we use the following function:

    crf_loss(c::CRF, input_seq, label_sequence, init_α)

```@example crf
using Flux: onehot
label_seq1 = [onehot(1, 1:2), onehot(1, 1:2), onehot(1, 1:2)]
label_seq2 = [onehot(1, 1:2), onehot(1, 1:2), onehot(2, 1:2)]
label_seq3 = [onehot(2, 1:2), onehot(1, 1:2), onehot(1, 1:2)]
label_seq4 = [onehot(2, 1:2), onehot(2, 1:2), onehot(2, 1:2)]

crf_loss(c, input_seq, label_seq1, init_α)
```

```@example crf
crf_loss(c, input_seq, label_seq2, init_α)
```

```@example crf
crf_loss(c, input_seq, label_seq3, init_α)
```

```@example crf
crf_loss(c, input_seq, label_seq4, init_α)
```

We can decode this using Viterbi Decode:

    viterbi_decode(c::CRF, input_seq, init_α)

```@example crf
viterbi_decode(c, input_seq, init_α) # Gives the label_sequence with least loss
```

This algorithm decodes the label sequence with the lowest loss value in polynomial time.

Currently the Viterbi Decode only supports CPU arrays.
When working with GPU, use viterbi_decode as follows:

    viterbi_decode(cpu(c), cpu.(input_seq), cpu(init_α))

### Working with Flux layers

CRFs work smoothly with Flux layers:

```@example crf
using Flux: Dense
NUM_FEATURES = 20

# For working with Dense layers, we can use 1D vectors
input_seq_dense = [rand(NUM_FEATURES) for i in 1:SEQUENCE_LENGTH]
```

```@example crf
m1 = Dense(NUM_FEATURES, NUM_LABELS + 2)
loss1(input_seq, label_seq) = crf_loss(c, m1.(input_seq), label_seq, init_α) # loss for model m1
loss1(input_seq_dense,  [onehot(1, 1:2), onehot(1, 1:2), onehot(1, 1:2)])
```


Here is an example of CRF with recurrent neural network layers:

```@example crf
# For recurrent layers, we need 2D input matrices (features × sequence_position)
# Let's create properly formatted 2D data
input_2d = [Float32.(rand(2, 1)) for i in 1:SEQUENCE_LENGTH]  # 2 features, 1 time step each
input_2d
```

```@example crf
using Flux: RNN
# Create a simple RNN model that works with 2D input
rnn_model = RNN(2 => 5)  # 2 input features → 5 hidden units
dense_layer = Dense(5, NUM_LABELS + 2)  # 5 hidden → 4 output (NUM_LABELS + 2)

# Forward pass through RNN then Dense layer
rnn_outputs = rnn_model.(input_2d)
final_outputs = dense_layer.(rnn_outputs)

# Now we can use this with CRF
loss_rnn(input_2d, label_seq) = crf_loss(c, dense_layer.(rnn_model.(input_2d)), label_seq, init_α)
loss_rnn(input_2d, [onehot(1, 1:2), onehot(2, 1:2), onehot(1, 1:2)])
```
