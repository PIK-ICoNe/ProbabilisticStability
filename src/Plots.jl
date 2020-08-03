
## WIP draw phase space pictures
function basin_plot(sol, x0, y0, in_basin; periodic_dim = 1, kwargs...)
    n = length(sol(0)) ÷ 2
    
    if periodic_dim == 1
        vars = [
            ((x, y) -> (mod2pi(x + π) - π, y), p + n, p)
            for p in findall(in_basin)
        ]
    elseif periodic_dim == 2
        vars = [
            ((x, y) -> (mod2pi(x + π) - π, y), p + n, p)
            for p in findall(in_basin)
        ]
    elseif isa(periodic_dim, Colon)
        vars = [
            ((x, y) -> (mod2pi(x + π) - π, mod2pi(y + π) - π), p + n, p)
            for p in findall(in_basin)
        ]
    else
        vars = [(p + n, p) for p in findall(in_basin)]
    end

    p = scatter(
        sol,
        vars = vars,
        shape = :rect,
        legend = false,
        c = :black,
        ms = 1,
        grid = false,
        markerstrokewidth = 0;
        kwargs...
    )
    if sum(in_basin) < n
        scatter!(
            p,
            sol,
            vars = [
                ((x, y) -> (mod2pi(x + π) - π, y), p + n, p)
                for p in findall(.!in_basin)
            ],
            shape = :rect,
            legend = false,
            c = :orange,
            ms = 1,
            grid = false,
            markerstrokewidth = 0;
            # bg_color = :red,
            kwargs...
        )
    end
    xlims!(extrema(x0)...)
    ylims!(extrema(y0)...)
    return p
end