# Named Entity Recognition

The API provided is a pretrained model for tagging Named Entities.
The current model supports 4 types of Named Entities:

- `PER`: Person
- `LOC`: Location
- `ORG`: Organisation
- `MISC`: Miscellaneous
- `O`: Not a Named Entity

To use the API, we first load the model weights into an instance of the tagger.
The function also accepts the path of model_weights and model_dicts (for character and word embeddings):

    NERTagger()
    NERTagger(dicts_path, weights_path)

```julia
julia> ner = NERTagger()
```
!!! note
    When you call `NERTagger()` for the first time, the package will request permission to download the `Model_dicts` and `Model_weights`. Upon downloading, these are stored locally and managed by `DataDeps`. So, on subsequent uses the weights will not need to be downloaded again.

Once we create an instance, we can call it to tag a String (sentence), sequence of tokens, `AbstractDocument` or `Corpus`.

    (ner::NERTagger)(sentence::String)
    (ner::NERTagger)(tokens::Vector{String})
    (ner::NERTagger)(sd::StringDocument)
    (ner::NERTagger)(fd::FileDocument)
    (ner::NERTagger)(td::TokenDocument)
    (ner::NERTagger)(crps::Corpus)

```julia
julia> sentence = "This package is maintained by John Doe."
"This package is maintained by John Doe."

julia> tags = ner(sentence)
8-element Vector{String}:
 "O"
 "O"
 "O"
 "O"
 "O"
 "PER"
 "PER"
 "O"

```

The API tokenizes the input sentences via the default tokenizer provided by `WordTokenizers`, this currently being set to the multilingual `TokTok Tokenizer`.

```
julia> using WordTokenizers

julia> collect(zip(WordTokenizers.tokenize(sentence), tags))
8-element Vector{Tuple{String,String}}:
 ("This", "O")
 ("package", "O")
 ("is", "O")
 ("maintained", "O")
 ("by", "O")
 ("John", "PER")
 ("Doe", "PER")
 (".", "O")

```

For tagging a multi-sentence text or document, one can use `split_sentences` from `WordTokenizers.jl` package and run the NER model on each.

```julia
julia> sentences = "Rabinov is winding up his term as ambassador. He will be replaced by Eliahu Ben-Elissar, a former Israeli envoy to Egypt and right-wing Likud party politician." # Sentence taken from CoNLL 2003 Dataset

julia> splitted_sents = WordTokenizers.split_sentences(sentences)

julia> tag_sequences = ner.(splitted_sents)
2-element Vector{Vector{String}}:
 ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
 ["O", "O", "O", "O", "O", "PER", "PER", "O", "O", "O", "MISC", "O", "O", "LOC", "O", "O", "ORG", "ORG", "O", "O"]

julia> zipped = [collect(zip(WordTokenizers.tokenize(splitted_sents[i]), tag_sequences[i])) for i in eachindex(splitted_sents)]

julia> zipped[1]
9-element Vector{Tuple{String,String}}:
 ("Rabinov", "O")
 ("is", "O")
 ("winding", "O")
 ("up", "O")
 ("his", "O")
 ("term", "O")
 ("as", "O")
 ("ambassador", "O")
 (".", "O")

julia> zipped[2]
20-element Vector{Tuple{String,String}}:
 ("He", "O")
 ("will", "O")
 ("be", "O")
 ("replaced", "O")
 ("by", "O")
 ("Eliahu", "PER")
 ("Ben-Elissar", "PER")
 (",", "O")
 ("a", "O")
 ("former", "O")
 ("Israeli", "MISC")
 ("envoy", "O")
 ("to", "O")
 ("Egypt", "LOC")
 ("and", "O")
 ("right-wing", "O")
 ("Likud", "ORG")
 ("party", "ORG")
 ("politician", "O")
 (".", "O")
```

Since tagging the Named Entities is done on sentence level,
the text of `AbstractDocument` is sentence-tokenized and then labelled for each sentence.
However, this is not possible for `NGramDocument` as text cannot be recreated.
For `TokenDocument`, text is approximated for splitting into sentences, hence the following throws a warning when tagging the `Corpus`.

```julia
julia> crps = Corpus([StringDocument("We aRE vErY ClOSE tO ThE HEaDQuarTeRS."), TokenDocument("this is Bangalore.")])
A Corpus with 2 documents:
 * 1 StringDocument's
 * 0 FileDocument's
 * 1 TokenDocument's
 * 0 NGramDocument's

Corpus's lexicon contains 0 tokens
Corpus's index contains 0 tokens

julia> ner(crps)
┌ Warning: TokenDocument's can only approximate the original text
└ @ TextAnalysis ~/.julia/dev/TextAnalysis/src/document.jl:220
2-element Vector{Vector{Vector{String}}}:
 [["O", "O", "O", "O", "O", "O", "O", "O"]]
 [["O", "O", "LOC", "O"]]
```
