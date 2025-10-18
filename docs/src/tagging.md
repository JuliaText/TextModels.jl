## Tagging Schemes

There are many tagging schemes used for sequence labelling.
TextAnalysis currently offers functions for conversion between these tagging formats.

*   BIO1
*   BIO2
*   BIOES

```julia
julia> using TextAnalysis
julia> tags = ["I-LOC", "O", "I-PER", "B-MISC", "I-MISC", "B-PER", "I-PER", "I-PER"]

julia> tag_scheme!(tags, "BIO1", "BIOES")

julia> tags
8-element Vector{String}:
 "S-LOC"
 "O"
 "S-PER"
 "B-MISC"
 "E-MISC"
 "B-PER"
 "I-PER"
 "E-PER"
```

## Parts of Speech Tagging

This package provides two different Part of Speech Taggers.

## Average Perceptron Part of Speech Tagger

This tagger can be used to find the POS tag of a word or token in a given sentence. It is based on the `Average Perceptron Algorithm`.
The model can be trained from scratch and weights are saved in a specified location.
The pretrained model can also be loaded and used directly to predict tags.

### To Train Model:
```julia
julia> tagger = TextModels.PerceptronTagger(false) #we can use tagger = TextModels.PerceptronTagger()
julia> fit!(tagger, [[("today","NN"),("is","VBZ"),("good","JJ"),("day","NN")]])
iteration : 1
iteration : 2
iteration : 3
iteration : 4
iteration : 5
```

### To Load Pretrained Model:
```julia
julia> tagger = TextModels.PerceptronTagger(true)
loaded successfully
PerceptronTagger(AveragePerceptron(Set(Any["JJS", "NNP_VBZ", "NN_NNS", "CC", "NNP_NNS", "EX", "NNP_TO", "VBD_DT", "LS", ("Council", "NNP")  …  "NNPS", "NNP_LS", "VB", "NNS_NN", "NNP_SYM", "VBZ", "VBZ_JJ", "UH", "SYM", "NNP_NN", "CD"]), Dict{Any,Any}("i+2 word wetlands"=>Dict{Any,Any}("NNS"=>0.0,"JJ"=>0.0,"NN"=>0.0),"i-1 tag+i word NNP basic"=>Dict{Any,Any}("JJ"=>0.0,"IN"=>0.0),"i-1 tag+i word DT chloride"=>Dict{Any,Any}("JJ"=>0.0,"NN"=>0.0),"i-1 tag+i word NN choo"=>Dict{Any,Any}("NNP"=>0.0,"NN"=>0.0),"i+1 word antarctica"=>Dict{Any,Any}("FW"=>0.0,"NN"=>0.0),"i-1 tag+i word -START- appendix"=>Dict{Any,Any}("NNP"=>0.0,"NNPS"=>0.0,"NN"=>0.0),"i-1 word wahoo"=>Dict{Any,Any}("JJ"=>0.0,"VBD"=>0.0),"i-1 tag+i word DT children's"=>Dict{Any,Any}("NNS"=>0.0,"NN"=>0.0),"i word dnipropetrovsk"=>Dict{Any,Any}("NNP"=>0.003,"NN"=>-0.003),"i suffix hla"=>Dict{Any,Any}("JJ"=>0.0,"NN"=>0.0)…), DefaultDict{Any,Any,Int64}(), DefaultDict{Any,Any,Int64}(), 1, ["-START-", "-START2-"]), Dict{Any,Any}("is"=>"VBZ","at"=>"IN","a"=>"DT","and"=>"CC","for"=>"IN","by"=>"IN","Retrieved"=>"VBN","was"=>"VBD","He"=>"PRP","in"=>"IN"…), Set(Any["JJS", "NNP_VBZ", "NN_NNS", "CC", "NNP_NNS", "EX", "NNP_TO", "VBD_DT", "LS", ("Council", "NNP")  …  "NNPS", "NNP_LS", "VB", "NNS_NN", "NNP_SYM", "VBZ", "VBZ_JJ", "UH", "SYM", "NNP_NN", "CD"]), ["-START-", "-START2-"], ["-END-", "-END2-"], Any[])
```

### To Predict Tags:

The perceptron tagger can predict tags over various document types:

    predict(tagger, sentence::String)
    predict(tagger, Tokens::Vector{String})
    predict(tagger, sd::StringDocument)
    predict(tagger, fd::FileDocument)
    predict(tagger, td::TokenDocument)

This can also be done by:
    tagger(input)


```julia
julia> predict(tagger, ["today", "is"])
2-element Vector{Any}:
 ("today", "NN")
 ("is", "VBZ")

julia> tagger(["today", "is"])
2-element Vector{Any}:
 ("today", "NN")
 ("is", "VBZ")
```

`PerceptronTagger(load::Bool)`

* `load`      - Boolean argument; if `true` then pretrained model is loaded

`fit!(self::PerceptronTagger, sentences::Vector{Vector{Tuple{String, String}}}, save_loc::String, nr_iter::Integer)`

* `self`      - `PerceptronTagger` object
* `sentences` - `Vector` of `Vector` of `Tuple` of pairs of word or token and its POS tag [see above example]
* `save_loc`  - location of file to save the trained weights
* `nr_iter`   - Number of iterations to pass the `sentences` to train the model (default 5)

`predict(self::PerceptronTagger, tokens)`

* `self`      - PerceptronTagger
* `tokens`    - `Vector` of words or tokens for which to predict tags

## Neural Model for Part of Speech Tagging Using LSTMs, CNN and CRF

The API provided is a pretrained model for tagging Part of Speech.
The current model tags all POS based on the [convention used in Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html), with 36 different Part of Speech tags excluding punctuation.

To use the API, we first load the model weights into an instance of the tagger.
The function also accepts the path of model_weights and model_dicts (for character and word embeddings):

    PoSTagger()
    PoSTagger(dicts_path, weights_path)

```julia
julia> pos = PoSTagger()

```

!!! note
    When you call `PoSTagger()` for the first time, the package will request permission to download the `Model_dicts` and `Model_weights`. Upon downloading, these are stored locally and managed by `DataDeps`. So, on subsequent uses the weights will not need to be downloaded again.

Once we create an instance, we can call it to tag a String (sentence), sequence of tokens, `AbstractDocument` or `Corpus`.

    (pos::PoSTagger)(sentence::String)
    (pos::PoSTagger)(tokens::Vector{String})
    (pos::PoSTagger)(sd::StringDocument)
    (pos::PoSTagger)(fd::FileDocument)
    (pos::PoSTagger)(td::TokenDocument)
    (pos::PoSTagger)(crps::Corpus)

```julia

julia> sentence = "This package is maintained by John Doe."
"This package is maintained by John Doe."

julia> tags = pos(sentence)
8-element Vector{String}:
 "DT"
 "NN"
 "VBZ"
 "VBN"
 "IN"
 "NNP"
 "NNP"
 "."

```

The API tokenizes the input sentences via the default tokenizer provided by `WordTokenizers`, this currently being set to the multilingual `TokTok Tokenizer`.

```

julia> using WordTokenizers

julia> collect(zip(WordTokenizers.tokenize(sentence), tags))
8-element Vector{Tuple{String,String}}:
 ("This", "DT")
 ("package", "NN")
 ("is", "VBZ")
 ("maintained", "VBN")
 ("by", "IN")
 ("John", "NNP")
 ("Doe", "NNP")
 (".", ".")

```

For tagging a multi-sentence text or document, one can use `split_sentences` from `WordTokenizers.jl` package and run the POS model on each.

```julia
julia> sentences = "Rabinov is winding up his term as ambassador. He will be replaced by Eliahu Ben-Elissar, a former Israeli envoy to Egypt and right-wing Likud party politician." # Sentence taken from CoNLL 2003 Dataset

julia> splitted_sents = WordTokenizers.split_sentences(sentences)

julia> tag_sequences = pos.(splitted_sents)
2-element Vector{Vector{String}}:
 ["CD", "VBZ", "VBG", "RP", "PRP\$", "NN", "IN", "NN", "."]
 ["PRP", "MD", "VB", "VBN", "IN", "NNP", "NNP", ",", "DT", "JJ", "JJ", "NN", "TO", "NNP", "CC", "JJ", "NNP", "NNP", "NNP", "."]

julia> zipped = [collect(zip(WordTokenizers.tokenize(splitted_sents[i]), tag_sequences[i])) for i in eachindex(splitted_sents)]

julia> zipped[1]
9-element Vector{Tuple{String,String}}:
 ("Rabinov", "CD")
 ("is", "VBZ")
 ("winding", "VBG")
 ("up", "RP")
 ("his", "PRP\$")
 ("term", "NN")
 ("as", "IN")
 ("ambassador", "NN")
 (".", ".")

julia> zipped[2]
20-element Vector{Tuple{String,String}}:
 ("He", "PRP")
 ("will", "MD")
 ("be", "VB")
 ("replaced", "VBN")
 ("by", "IN")
 ("Eliahu", "NNP")
 ("Ben-Elissar", "NNP")
 (",", ",")
 ("a", "DT")
 ("former", "JJ")
 ("Israeli", "JJ")
 ("envoy", "NN")
 ("to", "TO")
 ("Egypt", "NNP")
 ("and", "CC")
 ("right-wing", "JJ")
 ("Likud", "NNP")
 ("party", "NNP")
 ("politician", "NNP")
 (".", ".")

```

Since Part of Speech tagging is done on sentence level,
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

julia> pos(crps)
┌ Warning: TokenDocument's can only approximate the original text
└ @ TextAnalysis ~/.julia/dev/TextAnalysis/src/document.jl:220
2-element Vector{Vector{Vector{String}}}:
 [["PRP", "VBP", "RB", "JJ", "TO", "DT", "NN", "."]]
 [["DT", "VBZ", "NNP", "."]]

```
