using Documenter, TextModels

makedocs(
    modules = [TextModels],
    sitename = "TextModels",
    format = Documenter.HTML(
    ),
    pages = [
        "Home" => "index.md",
        "Conditional Random Fields" => "crf.md",
        "ULMFiT" => "ULMFiT.md",
        "Named Entity Recognition" => "ner.md",
        "Tagging Schemes" => "tagging.md",
        "Sentiment Analyzer" => "sentiment.md",
        "API References" => "APIReference.md"
    ],
)

