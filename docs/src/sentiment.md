## Sentiment Analyzer

It can be used to find the sentiment score (between 0 and 1) of a word, sentence or a Document.
A trained model (using Flux) on IMDB word corpus with weights saved are used to calculate the sentiments.

    model = SentimentAnalyzer()
    model(doc)
    model(doc, handle_unknown)

*  doc              = Input Document for calculating document (AbstractDocument type)
*  handle_unknown   = A function for handling unknown words. Should return an array (default (x)->[])

```julia
julia> using TextAnalysis

julia> m = SentimentAnalyzer()
Sentiment Analysis Model Trained on IMDB with a 88587 word corpus

julia> d1 = StringDocument("a very nice thing that everyone likes")
A StringDocument{String}
 * Language: Languages.English()
 * Title: Untitled Document
 * Author: Unknown Author
 * Timestamp: Unknown Time
 * Snippet: a very nice thing that everyone likes

julia> m(d1)
0.5183109f0

julia> d = StringDocument("a horrible thing that everyone hates")
A StringDocument{String}
 * Language: Languages.English()
 * Title: Untitled Document
 * Author: Unknown Author
 * Timestamp: Unknown Time
 * Snippet: a horrible thing that everyone hates

julia> m(d2)
0.47193584f0

```
