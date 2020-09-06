using Documenter, TextModels

makedocs(
    modules = [TextModels],
    sitename = "TextAnalysis",
    format = Documenter.HTML(
    ),
    pages = [
        "Home" => "index.md",
        "Conditional Random Fields" => "crf.md",
        "Named Entity Recognition" => "ner.md",
        "ULMFiT" => "ULMFiT.md",
        "API References" => "APIReference.md"
    ],
)

