# TextModels

A Julia package for natural language neural network models.

[![](https://github.com/JuliaText/TextModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaText/TextModels.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliahub.com/docs/TextModels)

> **Warning**
> The models in this repo are no longer state of the art -- the field has moved on very quickly. See [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) for more modern methods.  

## Introduction

The TextModels package enhances the TextAnalysis package with end-user focussed, practical natural language models, typically based on neural networks (in this case, [Flux](https://fluxml.ai/)).  Please see the [documentation](https://juliahub.com/docs/TextModels) for more.

- **License** : [MIT License](https://github.com/JuliaText/TextAnalysis.jl/blob/master/LICENSE.md)

## Installation

```julia
pkg> add TextModels
```

Some of the models require data files to run, which are downloaded on demand. Therefore, internet access is required at runtime for certain functionality in this package. 

## Contributing and Reporting Bugs

Contributions, in the form of bug-reports, pull requests, additional documentation are encouraged. They can be made to the Github repository.

**All contributions and communications should abide by the [Julia Community Standards](https://julialang.org/community/standards/).**

## Support

Feel free to ask for help on the [Julia Discourse forum](https://discourse.julialang.org/), or in the `#natural-language` channel on [julia-slack](https://julialang.slack.com). (Which you can [join here](https://slackinvite.julialang.org/)). You can also raise issues in this repository to request new features and/or improvements to the documentation and codebase.

