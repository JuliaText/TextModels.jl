using BSON

const PoSCharUNK = '¿'
const PoSWordUNK = "<UNK>"

struct PoSModel{M}
    model::M
end

"""
    PoSTagger()
    PoSTagger(dicts_path, weights_path)

Creates a Part-of-Speech tagger using a pretrained BiLSTM-CNN-CRF model.

The model performs POS tagging based on the Penn Treebank convention with 36 
different Part-of-Speech tags (excluding punctuation).

# Arguments
- `dicts_path::String`: Path to model dictionaries (character and word embeddings)
- `weights_path::String`: Path to model weights

# Usage
```julia
julia> pos = PoSTagger()  # Downloads pretrained model on first use
julia> pos("This package is maintained by John Doe.")
8-element Vector{String}:
 "DT"   # This
 "NN"   # package
 "VBZ"  # is
 "VBN"  # maintained
 "IN"   # by
 "NNP"  # John
 "NNP"  # Doe
 "."    # .
```

!!! note
    When calling `PoSTagger()` for the first time, the package will download 
    the model dictionaries and weights. These are stored locally and managed 
    by DataDeps for subsequent uses.
"""
PoSTagger() = PoSTagger(datadep"POS Model Dicts", datadep"POS Model Weights")

function PoSTagger(dicts_path, weights_path)
    labels, chars_idx, words_idx = load_model_dicts(dicts_path, false)
    model = BiLSTM_CNN_CRF_Model(labels, chars_idx, words_idx, chars_idx[PoSCharUNK], words_idx[PoSWordUNK], weights_path)
    PoSModel(model)
end

function (a::PoSModel)(tokens::Array{String,1})
    input_oh = [onehotinput(a.model, token) for token in tokens]
    return (a.model)(input_oh)
end

function (a::PoSModel)(sentence::AbstractString)
    a(WordTokenizers.tokenize(sentence))
end

function (a::PoSModel)(doc::AbstractDocument)
    return vcat(a.(WordTokenizers.split_sentences(text(doc))))
end

function (a::PoSModel)(ngd::NGramDocument)
    throw("Sequence Labelling not possible for NGramsDocument")
end

function (a::PoSModel)(crps::Corpus)
    return a.(crps.documents)
end
