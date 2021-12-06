using Test
using TextAnalysis
using TextModels

println("Running tests:")

include("crf.jl")
include("ner.jl")
include("pos.jl")
include("sentiment.jl")
include("averagePerceptronTagger.jl")
include("ulmfit.jl")
