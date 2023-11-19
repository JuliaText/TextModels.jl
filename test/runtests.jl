using Test
using TextAnalysis
using TextModels

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

tests = [
    "crf.jl"
    "ner.jl"
    "pos.jl"
    "sentiment.jl"
    "averagePerceptronTagger.jl"
    "ulmfit.jl"
]

function run_tests()
    for test in tests
        @info "Test: $test"
        Test.@testset verbose = true "\U1F4C2 $test" begin
            include(test)
        end
    end
end

@static if VERSION >= v"1.7"
    Test.@testset verbose = true showtiming = true "All tests" begin
        run_tests()
    end
else
    Test.@testset verbose = true begin
        run_tests()
    end
end
