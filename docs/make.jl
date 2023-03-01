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
        "ALBERT" => "ALBERT.md"
        "Pretraining Tutorial (ALBERT)" => "Pretraining_Tutorial(ALBERT).md",
        "Finetuning Tutorial (ALBERT)" => "Training_tutorial.md"
        "API References" => "APIReference.md"
    ],
)

