# Conditional Random Fields

This package currently provides support for Linear Chain Conditional Random Fields.

Let us first load the dependencies-

    using Flux
    using Flux: onehot, train!, Params, gradient, LSTM, Dense, reset!
    using TextModels: CRF, viterbi_decode, crf_loss

Conditional Random Field layer is essentially like a softmax that operates on the top most layer.

Let us suppose the following input sequence to the CRF with `NUM_LABELS = 2`

```julia
julia> NUM_LABELS = 2
julia> SEQUENCE_LENGTH = 2 # CRFs can handle variable length inputs sequences
julia> input_seq = [Float32.(rand(NUM_LABELS + 2)) for i in 1:SEQUENCE_LENGTH] # NUM_LABELS + 2, where two extra features correspond to the :START and :END label.
2-element Vector{Vector{Float32}}:
 [0.5114323, 0.5355139, 0.4011792, 0.56359255]
 [0.22925346, 0.21232551, 0.77616125, 0.41560093]

```

We define our crf layer as -

    CRF(NUM_LABELS::Integer)

```julia
julia> c = CRF(NUM_LABELS) # The API internally append the START and END tags to NUM_LABELS.
CRF with 4 distinct tags (including START and STOP tags).
```

Now as for the initial variable in Viterbi Decode or Forward Algorithm,
we define our input as

```julia
julia>  init_α = fill(-10000, (c.n + 2, 1))
julia>  init_α[c.n + 1] = 0
```

Optionally this could be shifted to GPU by `init_α = gpu(init_α)`,
considering the input sequence to be CuArray in this case.
To shift a CRF `c` to gpu, one can use `c = gpu(c)`.

To find out the crf loss, we use the following function -

    crf_loss(c::CRF, input_seq, label_sequence, init_α)

```
julia> label_seq1 = [onehot(1, 1:2), onehot(1, 1:2)]

julia> label_seq2 = [onehot(1, 1:2), onehot(2, 1:2)]

julia> label_seq3 = [onehot(2, 1:2), onehot(1, 1:2)]

julia> label_seq4 = [onehot(2, 1:2), onehot(2, 1:2)]

julia> crf_loss(c, input_seq, label_seq1, init_α)
1.33554f0

julia> crf_loss(c, input_seq, label_seq2, init_α)
1.2327178f0

julia> crf_loss(c, input_seq, label_seq3, init_α)
1.3454239f0

julia> crf_loss(c, input_seq, label_seq4, init_α)
1.6871009f0

```

We can decode this using Viterbi Decode.

    viterbi_decode(c::CRF, input_seq, init_α)

```julia
julia> viterbi_decode(c, input_seq, init_α) # Gives the label_sequence with least loss
2-element Vector{Flux.OneHotArray{UInt32, 2, 0, 1, UInt32}}:
 [1, 0]
 [0, 1]

```

This algorithm decodes for the label sequence with lowest loss value in polynomial time.

Currently the Viterbi Decode only support cpu arrays.
When working with GPU, use viterbi_decode as follows

    viterbi_decode(cpu(c), cpu.(input_seq), cpu(init_α))

### Working with Flux layers

CRFs smoothly work over Flux layers-

```julia
julia> NUM_FEATURES = 20

julia> input_seq = [rand(NUM_FEATURES) for i in 1:SEQUENCE_LENGTH]
2-element Vector{Vector{Float32}}:
 [0.948219, 0.719964, 0.352734, 0.0677656, 0.570564, 0.187673, 0.525125, 0.787807, 0.262452, 0.472472, 0.573259, 0.643369, 0.00592054, 0.945258, 0.951466, 0.323156, 0.679573, 0.663285, 0.218595, 0.152846]
 [0.433295, 0.11998, 0.99615, 0.530107, 0.188887, 0.897213, 0.993726, 0.0799431, 0.953333, 0.941808, 0.982638, 0.0919345, 0.27504, 0.894169, 0.66818, 0.449537, 0.93063, 0.384957, 0.415114, 0.212203]

julia> m1 = Dense(NUM_FEATURES, NUM_LABELS + 2)

julia> loss1(input_seq, label_seq) = crf_loss(c, m1.(input_seq), label_seq, init_α) # loss for model m1

julia> loss1(input_seq,  [onehot(1, 1:2), onehot(1, 1:2)])
4.6620379898687485

```


Here is an example of CRF with LSTM and Dense layer -

```julia
julia> LSTM_SIZE = 10

julia> lstm = LSTM(NUM_FEATURES, LSTM_SIZE)

julia> dense_out = Dense(LSTM_SIZE, NUM_LABELS + 2)

julia> m2(x) = dense_out.(lstm.(x))

julia> loss2(input_seq, label_seq) = crf_loss(c, m2(input_seq), label_seq, init_α) # loss for model m2

julia> loss2(input_seq,  [onehot(1, 1:2), onehot(1, 1:2)])
1.6501050910529504

julia> reset!(lstm)
```
