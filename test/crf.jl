using Flux
using Flux: gradient, LSTM, Dense, reset!, onehot, onehotbatch, RNN, params
using TextModels: score_sequence, forward_score
using DelimitedFiles
using LinearAlgebra

@testset "crf" begin

    ks = [onehotbatch([1, 1, 1], 1:2),
          onehotbatch([1, 1, 2], 1:2),
          onehotbatch([1, 2, 1], 1:2),
          onehotbatch([1, 2, 2], 1:2),
          onehotbatch([2, 1, 1], 1:2),
          onehotbatch([2, 1, 2], 1:2),
          onehotbatch([2, 2, 1], 1:2),
          onehotbatch([2, 2, 2], 1:2)]

    @testset "Loss function" begin

        c = CRF(2)
        input_seq = rand(4, 3)

        scores = [score_sequence(c, input_seq, k) for k in ks]

        init_α = fill(-10000, (c.n + 2, 1))
        init_α[c.n + 1] = 0

        s1 = sum(exp.(scores))

        s2 = exp(forward_score(c, input_seq, init_α))

        @test abs(s1 - s2) / max(s1, s2) <= 0.00000001
    end

    @testset "Viterbi Decode" begin
        c = CRF(2)
        input_seq = rand(4, 3)

        scores = [score_sequence(c, input_seq, k) for k in ks]

        maxscore_idx = argmax(scores)

        init_α = fill(-10000, (c.n + 2, 1))
        init_α[c.n + 1] = 0

        @test viterbi_decode(c, input_seq, init_α) == ks[maxscore_idx]
    end

    @testset "CRF with Flux Layers" begin
        path = "data/weather.csv"

        function load(path::String)
            m = readdlm(path, ',')
            n, nf = size(m)
            Ys = m[:, end]
            ls = unique(Ys)
            nl = length(ls)
            return Matrix{Float32}(m[:, 1:2]'), onehotbatch(Ys, ls), n, nf-1, ls, nl
        end

        X, Y, n, num_features, labels, num_labels = load(path)

        _, T = size(X)

        LSTM_STATE_SIZE = 5

        m = Chain(RNN(num_features, LSTM_STATE_SIZE), Dense(LSTM_STATE_SIZE, num_labels + 2))

        c = CRF(num_labels)

        init_α = fill(-10000, (c.n + 2, 1))
        init_α[c.n + 1] = 0

        opt = Descent(0.01)

        ps = params(params(m)..., params(c)...)

        NBATCH = 15

        loss(xs, ys) = crf_loss(c, m(xs), ys, init_α) + 1e-4*sum(c.W.*c.W)

        function train()
            i = 1
            while true
                l = i + NBATCH > T ? T : (i + NBATCH - 1)
                xbatch, ybatch = (@view X[:, i:l]), (@view Y[:, i:l])
                reset!(m[1])
                grads = Flux.gradient(() -> loss(xbatch, ybatch), ps)
                Flux.Optimise.update!(opt, ps, grads)
                i += NBATCH
                i >= T && break
            end
        end

        function find_loss(x, y)
            reset!(m[1])
            loss(x, y)
        end

        l1 = find_loss(X, Y)
        dense_param_1 = deepcopy(m[2].weight)
        lstm_param_1 = deepcopy(m[1].cell.Wh)
        crf_param_1 = deepcopy(c.W)

        for i in 1:10
            train()
        end

        dense_param_2 = deepcopy(m[2].weight)
        lstm_param_2 = deepcopy(m[1].cell.Wh)
        crf_param_2 = deepcopy(c.W)
        l2 = find_loss(X, Y)

        @test l1 > l2
        @test dense_param_1 != dense_param_2
        @test lstm_param_1 != lstm_param_2
        @test crf_param_1 != crf_param_2
    end
end
