using BSON

const PoSCharUNK = '¿'
const PoSWordUNK = "<UNK>"

struct PoSModel{M}
    model::M
end

PoSTagger() = PoSTagger(datadep"POS Model Dicts", datadep"POS Model Weights")

function PoSTagger(dicts_path, weights_path)
    labels, chars_idx, words_idx = load_model_dicts(dicts_path, false)
    model = BiLSTM_CNN_CRF_Model(labels, chars_idx, words_idx, chars_idx[PoSCharUNK], words_idx[PoSWordUNK], weights_path)
    PoSModel(model)
end

(a::PoSModel)(tokens::Array{String,1}) = (a.model)(onehotinput(a.model, tokens))

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
