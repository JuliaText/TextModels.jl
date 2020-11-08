## Preface

The TextModels package enhances the TextAnalysis package with end-user focussed, practical natural language models, typically based on neural networks (in this case, [Flux](https://fluxml.ai/))

This package depends on the [TextAnalysis](https://github.com/JuliaText/TextAnalysis.jl) package, which contains basic algorithms to deal with textual documetns. 

## Installation

The TextModels package can be installed using Julia's package manager:

    Pkg.add("TextModels")

Some of the models require data files to run, which are downloaded on demand. Therefore, internet access is required at runtime for certain functionality in this package. 
