"""
Linear Chain - CRF Layer.

For input sequence `x`,
predicts the most probable tag sequence `y`,
over the set of all possible tagging sequences `Y`.

In this CRF, two kinds of potentials are defined,
emission and Transition.
"""
mutable struct CRF{S}
    W::S        # Transition Scores
    n::Int      # Num Labels
end

"""
Second last index for start tag,
last one for stop tag .
"""
function CRF(n::Integer)
    W = rand(Float32, n + 2, n + 2)

    # Transition to start state should be penalized.
    W[:, n + 1] .= -10000
    # Transition out of stop state should be penalized.
    W[n + 2, :] .= -10000

    return CRF(W, n)
end

@functor CRF (W,)

function Base.show(io::IO, c::CRF)
    print(io, "CRF with ", c.n + 2, " distinct tags (including START and STOP tags).")
end

function (a::CRF)(x_seq, init_α)
    viterbi_decode(a, x_seq, init_α)
end
