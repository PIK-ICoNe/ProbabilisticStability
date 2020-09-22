
using Documenter
using ProbabilisticStability

makedocs(
    modules = [ProbabilisticStability],
    authors = "Paul Schultz",
    sitename = "ProbabilisticStability.jl",
    format = Documenter.HTML()

deploydocs(
    repo = "github.com/luap-pik/ProbabilisticStability.git",
    target = "build"
)
