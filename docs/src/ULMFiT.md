# ULMFiT

This is the implementation of the [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf) paper released by Jeremy Howard and Sebastian Ruder. The model can be used for several classification tasks in the Natural Language Processing domain. The model follows the concept of [Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning). Here, the model was trained to perform a Sentiment Analysis task. The weights for this are also provided, as well as the weights for the Language model part of ULMFiT, so that it can be used to fine-tune the model for different tasks.

## Data Loading and Preprocessing

Proper preprocessing is essential before starting training ULMFiT. For the pretraining step for Language model, a general-purpose corpus is needed, which here is WikiText-103 by default. Similarly, for fine-tuning Language Model and fine-tuning classifier we need a dataset for the specific task (example IMDB for Sentiment Analysis, large scale AG news and DBpedia ontology datasets for Topic classification etc). To load data for these steps, data loaders need to be defined. Since the data used to train such a large model is large, it is not recommended to load all the data at once, instead the data should be loaded in batches through the concept of tasks (or coroutines) in Julia (Refer to [this](https://docs.julialang.org/en/v1.0/manual/control-flow/#man-tasks-1) documentation for understanding tasks in Julia) using `Channels`. Basically, we need to create a `Channel` which supplies a mini-batch at every call. As an example, the functions used for preprocessing of the IMDB dataset are given in the `data_loaders.jl` in the ULMFiT directory. Also, for loading WikiText-103 dataset and IMDB dataset, default functions are provided in the same file.

Default data loaders are provided in the `data_loaders.jl`:

 * `load_wikitext_103`    : returns `Channel` which gives batches from WikiText-103 dataset
 * `imdb_fine_tune_data`  : returns `Channel` for loading fine-tuning data from IMDb movie review dataset
 * `imdb_classifier_data` : returns `Channel` for loading classification data from IMDB movie review dataset for binary sentiment analysis

 To make custom loaders, have a look at these functions. These will give a clear idea of the preparation of batches inside data loaders.

## Step 1 - Pretraining Language Model

In this step, the Language Model will learn the general properties of the language. To train the model we need a general domain corpus like WikiText-103. For training, a `generator` function is provided to create a `Channel` which will give a mini-batch in every call. After pre-processing the corpus, the tokenized corpus is given as input to the generator function and the Channel can be created like so:
```julia
julia> loader = ULMFiT.imdb_fine_tune_data(4, 10) # batchsize=4, bptt=10
Channel{Any}(0) (1 item available)

julia> max_batches = take!(loader) # this is the first call to the loader

# These are the subsequent calls in pairs for X and Y
julia> X = take!(loader)
10-element Vector{Vector{Any}}:
 ["i", "transparent", "it", "were"]
 ["admit", "villain", "immediately", "all"]
 [",", "who", "as", "first"]
 ["the", "talks", "she", "rate"]
 ["great", "like", "is", "."]
 ["majority", "mortimer", "on", "even"]
 ["of", "snerd", "for", "veda"]
 ["films", "and", "a", "ann"]
 ["released", "has", "few", "borg"]
 ["before", "an", "seconds", "in"]

julia> Y = take!(loader)
10-element Vector{Vector{Any}}:
 ["admit", "villain", "immediately", "all"]
 [",", "who", "as", "first"]
 ["the", "talks", "she", "rate"]
 ["great", "like", "is", "."]
 ["majority", "mortimer", "on", "even"]
 ["of", "snerd", "for", "veda"]
 ["films", "and", "a", "ann"]
 ["released", "has", "few", "borg"]
 ["before", "an", "seconds", "in"]
 ["say", "office", ",", "a"]

```
Note that at the first call to this `Channel` the output will be the maximum number of batches which it can give. Two calls to this `Channel` complete one batch, that is, it does not give `X` and `Y` both together in one call, two calls are needed: first `X` is given out and in the second call `Y`. Also, to understand what `batchsize` and `bptt` are, refer to this [blog](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-2).

### Training Language Model

The file `pretrain_lm.jl` contains the whole implementation of the `LanguageModel`. To start training, first create an instance of the `LanguageModel` type, then use the below specified function with appropriate arguments.

```julia
julia> lm = LanguageModel()
```

It has several arguments to define the internal structure of the `LanguageModel` instance:
[All are keyword arguments and optional]

 * `embedding_size`      : defines size of embeddings for embedding matrix in `DroppedEmbeddings` layer (default value is 400)
 * `hid_lstm_sz`         : defines size of hidden `AWD_LSTM` layer (default value is 1150)
 * `out_lstm_sz`         : defines size of output `AWD_LSTM` layer (default value is equal to `embedding_size`)
 * `embed_drop_prob`     : embedding dropout probability in `DroppedEmbeddings` (default value is 0.05)
 * `word_drop_prob`      : dropout probability to the input embeddings to first `AWD_LSTM` layer (default value is 0.4)
 * `hid_drop_prob`       : DropConnect probability to the hidden matrices of the each `AWD_LSTM` layer (default value is 0.5)
 * `layer_drop_prob`     : probability of the dropout layer between the `AWD_LSTM` layers (default value is 0.3)
 * `final_drop_prob`     : probability of the dropout layer after the last `AWD_LSTM` layer (default value is 0.3)


```julia
pretrain_lm!(lm::LanguageModel=LanguageModel(),
            data_loader::Channel=load_wikitext_103;
            base_lr=0.004,
            epochs::Integer=1,
            checkpoint_itvl::Integer=5000)
```

Positional Arguments:

 * `lm`               : instance of `LanguageModel` struct
 * `data_loader`      : this `Channel` is created to load the data from the general-domain corpus

Keyword Arguments:

 * `base_lr`          : learning rate for `ADAM` optimizer
 * `epochs`           : number of epochs
 * `checkpoint_itvl`  : Checkpoint interval, the number of iterations after which the model weights are saved to a specified BSON file

[All default values shown above]

To know the full implementation of the `LanguageModel`, `AWD_LSTM` layer and `DroppedEmbeddings` layer, refer [blog1](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-for-text-clasification) and [blog2](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-2).

## Step 2 - Fine-tuning Language Model

In this step, the Language Model pretrained in the last step will be fine-tuned on the target data of the downstream task (e.g. sentiment analysis). Again, preprocess the text data from the dataset and create a `Channel` using the `generator` function. The file `fine_tune_lm.jl` contains all the functions related to fine-tuning of the Language model.

### Fine-tune Language Model

The `fine_tune_lm!` function is used to fine-tune a Language Model:

```julia
fine_tune_lm!(lm::LanguageModel=load_lm(),
        data_loader::Channel=imdb_fine_tune_data,
        stlr_cut_frac::Float64=0.1,
        stlr_ratio::Float32=32,
        stlr_η_max::Float64=0.01;
        epochs::Integer=1,
        checkpoint_itvl::Integer=5000
)
```

Positional Arguments:

 * `lm`               : Instance of `LanguageModel` struct
 * `data_loader`      : `Channel` created to load mini-batches from target data

Keyword Arguments:

 * `stlr_cut_frac`    : In STLR, it is the fraction of iterations for which LR is increased
 * `stlr_ratio`       : In STLR, it specifies how much smaller is lowest LR from maximum LR
 * `stlr_η_max`       : In STLR, this is the maximum LR value
 * `epochs`           : the number of epochs for which the language model is to be fine-tuned
 * `checkpoint_itvl`  : Checkpoint interval, the number of iterations after which the model weights are saved to a specified BSON file

[All default values shown above]
By default the `fine_tune_lm!` function will load a pretrained model if a `LanguageModel` instance is not provided.

In the fine-tuning step, some additional techniques are used for training, namely, Discriminative fine-tuning and Slanted triangular learning rates (STLR). To know their implementation refer to [this](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-3) blog.

## Step 3 - Fine-tuning the classifier for downstream task

This is the final step of training the ULMFiT model for a specific task. Here, two linear blocks will be added to the Language model layers. These are `PooledDense` and `Dense`. To know more about them go through [this](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-5) blog post.

### Fine-tune text classifier

Before starting training, it is required to make an instance of the `TextClassifier` type like so:

```julia
julia> classifier = TextClassifier()
```

Arguments:
[All are positional and optional arguments]

  * `lm`                   : Instance of `LanguageModel` [by default `LanguageModel()`]
  * `clsfr_out_sz`         : output `Dense` layer size of classifier [default value is 2]
  * `clsfr_hidden_sz`      : hidden `PooledDense` layer size of classifier [default value is 50]
  * `clsfr_hidden_drop`    : dropout probability for the `PooledDense` layer [hidden layer] of classifier [default value is 0.4]

To start training, use the `train_classifier!` function:

```julia
train_classifier!(classifier::TextClassifier=TextClassifier(),
        classes::Integer=1,
        data_loader::Channel=imdb_classifier_data,
        hidden_layer_size::Integer=50;
        stlr_cut_frac::Float64=0.1,
        stlr_ratio::Number=32,
        stlr_η_max::Float64=0.01,
        val_loader::Channel=nothing,
        cross_val_batches::Union{Colon, Integer}=:,
        epochs::Integer=1,
        checkpoint_itvl=5000
)
```

Positional Arguments:

 * `lm`               : Instance of `LanguageModel` struct
 * `classes`          : Size of output layer for classifier or number of classes for which the classifier is to be trained
 * `data_loader`      : `Channel` created to load mini-batches for classification
 * `hidden_layer_size`: Size of the hidden linear layer added for making the classifier

Keyword Arguments:

 * `stlr_cut_frac`    : In STLR, it is the fraction of iterations for which LR is increased
 * `stlr_ratio`       : In STLR, it specifies how much smaller is lowest LR from maximum LR
 * `stlr_η_max`       : In STLR, this is the maximum LR value
 * `val_loader`       : `Channel` which will load the cross validation set as mini-batches same as `data_loader`
 * `cross_val_batches`: number of cross validation batches for which the accuracy and loss will be printed
 * `epochs`           : the number of epochs for which the language model is to be fine-tuned
 * `checkpoint_itvl`  : Checkpoint interval, the number of iterations after which the model weights are saved to a specified BSON file

[All default values are shown above]

## Layers

There are some custom layers added for this model to work properly. All of them are described below; go through all of them to have a better understanding of the model.

### Weight-Dropped LSTM (WeightDroppedLSTM)

This is basically a modification to the original LSTM layer. The layer uses [DropConnect](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf) with [Variational-dropping](https://arxiv.org/abs/1506.02557) concepts. In this approach, the hidden-to-hidden weights and input-to-hidden weights can be dropped randomly for a given probability. That means the layer uses the same drop mask for all timesteps and to do this, the layer saves the masks. To change the mask, the `reset_masks!` function should be used.

```julia
# maskWi and maskWh are drop masks for Wi and Wh weights
julia> fieldnames(ULMFiT.WeightDroppedLSTMCell)
(:Wi, :Wh, :b, :h, :c, :p, :maskWi, :maskWh, :active)

# To deine a layer with 4 input size and 5 output size and 0.3 dropping probability
julia> wd = ULMFiT.WeightDroppedLSTM(4, 5, 0.3);

# Pass
julia> x = rand(4);
julia> state = wd(x)
(([0.6103202972778569; 0.6103202972778569; 0.6103202972778569; 0.6103202972778569; 0.6103202972778569;;], [0.7274433138892508; 0.7274433138892508; 0.7274433138892508; 0.7274433138892508; 0.7274433138892508;;]), [0.6103202972778569; 0.6103202972778569; 0.6103202972778569; 0.6103202972778569; 0.6103202972778569;;])

# The output is a tuple containing ((hidden_state, cell_state), output)

# To reset_masks!
julia> ULMFiT.reset_masks!(wd)
```

### Averaged-SGD LSTM (AWD_LSTM)

This is a regular LSTM layer with Variational DropConnect and weights averaging functionality (while training). This layer comes out to be efficient for Language modelling tasks (refer [this](https://arxiv.org/pdf/1708.02182.pdf)). It used the `WeightDroppedLSTM` layer discussed above for DropConnect property. It averages the weights on subsequent iteration after trigger iteration. The layer needs a trigger iteration number to use its averaging functionality. To set the trigger `set_trigger!` function can be used and `reset_masks!` can be used for resetting drop masks for DropConnect.

```julia
# `accum` field is used to store the sum of weights for every iteration after trigger
# to get average of the weights for every subsequent iteration
julia> fieldnames(ULMFiT.AWD_LSTM)
(:layer, :T, :accum)

julia> awd = ULMFiT.AWD_LSTM(3, 4, 0.5)

# Setting trigger iteration
julia> ULMFiT.set_trigger!(1000, awd)
julia> awd.T
1000

# Pass
julia> x = rand(3);
julia> state = awd(x)
(([0.6103202972778569; 0.6103202972778569; 0.6103202972778569; 0.6103202972778569;;], [0.7274433138892508; 0.7274433138892508; 0.7274433138892508; 0.7274433138892508;;]), [0.6103202972778569; 0.6103202972778569; 0.6103202972778569; 0.6103202972778569;;])

 # Resetting drop masks
julia> awd.layer.cell.maskWi[1:5, :]
3×3 Matrix{Float32}:
 1.21938  1.21039  1.73037
 0.0      0.0      0.0
 0.0      1.35731  1.04278
 0.0      1.058    0.0
 0.0      1.1709   1.12383

julia> ULMFiT.reset_masks!(awd)
julia> awd.layer.cell.maskWi[1:5, :]
3×3 Matrix{Float32}:
 1.22731  0.0      1.95435
 0.0      1.28479  0.0
 0.0      1.51710  1.41950
 1.68891  0.0      0.0
 0.0      0.0      0.0
```

### Variational-DropOut (VarDrop)

This layer applies Variational-DropOut, which is using the same dropout mask until it is specified to change or until a pass is over. This dropout is useful for recurrent layers since these layers perform better if the same mask is used for all time-steps (pass) instead of using different masks for every timestep. [Refer to [this](https://arxiv.org/pdf/1506.02557.pdf) paper for more details]. This layer saves the masks after generation until it is specified to change. To change the mask use the `reset_masks!` function.

```julia
julia> vd = ULMFiT.VarDrop(0.5)
VarDrop{Float64}(0.5, Matrix{Float32}(undef, 0, 0), true, true)

# No mask generation when nothing is passed
julia> vd.mask
0×0 Matrix{Float32}

julia> x = rand(4,5)
4×5 Matrix{Float64}:
 0.383492  0.914917  0.616324  0.940116  0.526015
 0.286494  0.35078   0.320465  0.334261  0.295965
 0.232206  0.26289   0.940569  0.23259   0.675406
 0.152903  0.934304  0.125803  0.727792  0.239359

julia> x = vd(x)
4×5 Matrix{Float64}:
 0.0       0.0  0.0      1.88023   1.05203
 0.0       0.0  0.64093  0.668522  0.591929
 0.464413  0.0  1.88114  0.0       0.0
 0.0       0.0  0.0      0.0       0.478717

julia> vd.mask
4×5 Matrix{Float64}:
 0.0  0.0  0.0  2.0  2.0
 0.0  0.0  2.0  2.0  2.0
 2.0  0.0  2.0  0.0  0.0
 0.0  0.0  0.0  0.0  2.0

```

### Dropped Embeddings (DroppedEmbeddings)

This layer is an embedding layer which can work in two ways: either to give embedding vectors for the given indices of words in vocabulary or can be used to get probability distribution for all the words of vocabulary with a softmax layer, which is also called weight-tying. Here, it can be used to tie weights of the embedding layer and the last softmax layer. In addition to this, it also drops embeddings for words randomly for a given probability of dropping; in other words, it puts the whole embedding vector of randomly selected words to a vector of zeros. Here, the mask used for the dropping possesses a variational property, that is, it cannot be changed until it is specified to change or generate a new drop mask. `reset_masks!` should be used to reset the mask.

```julia
julia> fieldnames(ULMFiT.DroppedEmbeddings)
(:emb, :p, :mask, :active)

julia> de = ULMFiT.DroppedEmbeddings(5, 2, 0.3);

# Pass
julia> x = [4,2,1];
julia> embeddings = de(x)
2×3 transpose(::Matrix{Float32}) with eltype Float32:
 -0.0805887  0.0  0.0
 -0.147202  -0.0 -0.0

julia> de.mask
5-element Vector{Float32}:
 0.0
 0.0
 0.0
 0.87639743
 0.0

# reset mask
julia> reset_masks!(de)
julia> de.mask
5-element Vector{Float32}:
 0.0
 0.6529015
 0.0
 0.0
 0.5247689
```

### Concat-Pooled Dense layer

This is a simple modification to the original `Dense` layer for recurrent networks. This layer should come after last RNN layer in the network. It takes the `Vector` of outputs of the RNN layers at all timesteps and then performs max and mean pooling to those outputs, then concatenates these outputs with the last output of the RNN layers and passes this concatenation result to the a `Dense` layer within.

```julia
# The first argument is the length of the output Vector of the preceding RNN layer to this layer. Also, by default if uses identity activation, it can be changed by giving desired activaiton as the third argument
julia> pd = ULMFiT.PooledDense(4, 3)

# Pass
julia> X = [rand(4), rand(4), rand(4)];
julia> pd(X)
3×1 Matrix{Float64}:
 -1.3679283360573462
  1.1115990254044759
 -0.27398355913859046
```
