module TextModels

    using WordTokenizers
    using Languages

    using DataDeps
    using Pkg.Artifacts


    using Flux, Zygote
    using Flux: identity, onehot, onecold, @functor, onehotbatch


    using TextAnalysis
    import TextAnalysis: fit!, predict

    export CRF, viterbi_decode, crf_loss

    export NERTagger, PoSTagger, SentimentAnalyzer



    include("averagePerceptronTagger.jl")
    include("sentiment.jl")


    # CRF
    include("CRF/crf.jl")
    include("CRF/predict.jl")
    include("CRF/crf_utils.jl")
    include("CRF/loss.jl")

    # NER and POS
    include("sequence/ner_datadeps.jl")
    include("sequence/ner.jl")
    include("sequence/pos_datadeps.jl")
    include("sequence/pos.jl")
    include("sequence/sequence_models.jl")
    
    # ALBERT
    function __init__()
        include(joinpath(@__DIR__, "./albert/datadeps.jl"))
    end 
    include("./albert/ALBERT.jl")

    # ULMFiT
    module ULMFiT
        using TextAnalysis
        using Flux
        using Flux:crossentropy
        using Zygote
        using BSON
        using CorpusLoaders
        using DataDeps
        include("ULMFiT/utils.jl")
        include("ULMFiT/datadeps.jl")
        include("ULMFiT/data_loaders.jl")
        include("ULMFiT/custom_layers.jl")
        include("ULMFiT/pretrain_lm.jl")
        include("ULMFiT/fine_tune_lm.jl")
        include("ULMFiT/train_text_classifier.jl")
    end
    export ULMFiT

    function __init__()
        pos_tagger_datadep_register()
        ner_datadep_register()
        pos_datadep_register()
        ULMFiT.ulmfit_datadep_register()
    
        global sentiment_model = artifact"sentiment_model"
    end
end
