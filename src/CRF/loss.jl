"""
    forward_score(c::CRF, x, init_α)

Compute the normalization/partition function (Forward Algorithm score) `Z`.

This function calculates the total score of all possible label sequences
for the given input sequence, which is used as the denominator in the 
CRF probability computation.

# Arguments
- `c::CRF`: The CRF model
- `x`: Input sequence (emission scores)
- `init_α`: Initial alpha values for the forward algorithm

# Returns
- The log of the partition function `Z`
"""
function forward_score(c::CRF, x, init_α)
    forward_var = log_sum_exp(c.W .+ x[1]' .+ init_α)

    for i in 2:length(x)
        forward_var = log_sum_exp((c.W .+ x[i]') .+ forward_var')
    end

    return log_sum_exp(c.W[:, c.n + 2] .+ forward_var')[1]
end

"""
    score_sequence(c::CRF, x, label_seq)

Calculate the score of the desired `label_seq` against sequence `x`.

This function computes the unnormalized score (not exponentiated) for a given 
label sequence, which is required for the negative log likelihood calculation.

# Arguments
- `c::CRF`: The CRF model
- `x`: Input sequence (emission scores)
- `label_seq`: Target label sequence where `eltype(label_seq) = Flux.OneHotVector`

# Returns
- Score value for the given label sequence
"""
function score_sequence(c::CRF, x, label_seq)
    score = preds_first(c, label_seq[1]) + x[1][onecold(label_seq[1])]

    for i in 2:length(label_seq)
        score += preds_single(c, label_seq[i], label_seq[i-1]) +
                    x[i][onecold(label_seq[i])]
    end

    return score + preds_last(c, label_seq[end])
end

"""
    crf_loss(c::CRF, x, label_seq, init_α)

Compute the negative log-likelihood loss for CRF training.

This function calculates the CRF loss as the difference between the forward score
(log partition function) and the score of the true label sequence. This represents
the negative log-likelihood of the true sequence under the CRF model.

# Arguments
- `c::CRF`: The CRF model
- `x`: Input sequence (emission scores)
- `label_seq`: True label sequence
- `init_α`: Initial alpha values for the forward algorithm

# Returns
- The negative log-likelihood loss value

# Formula
```
loss = forward_score(c, x, init_α) - score_sequence(c, x, label_seq)
     = log(Z) - score(true_sequence)
```
"""
crf_loss(c::CRF, x, label_seq, init_α) = forward_score(c, x, init_α) - score_sequence(c, x, label_seq)
