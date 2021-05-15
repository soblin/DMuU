using Plots
using LinearAlgebra

mutable struct Grid2D
    xind::Int64
    yind::Int64
    value::Matrix{Float64}
    reward_::Matrix{Float64}
    terminals::Vector{NTuple{2, Int64}}
    action_sets::Vector{Symbol}
    π::Matrix{Vector{Symbol}} # in case several a's give same maximum value
    value1::Matrix{Float64}
    value2::Matrix{Float64}
    switch::Bool
    initial::Bool
end

function Grid2D(xind::Int64, yind::Int64, rewards::Dict{NTuple{2, Int64}, Float64}, terminals::Vector{NTuple{2, Int64}})
    value1 = zeros(xind, yind)
    value2 = zeros(xind, yind)
    reward = zeros(xind, yind)
    for (pos, r) in rewards
        reward[pos...] = r
    end
    π = fill([:N, :W, :S, :E], xind, yind)    
    return Grid2D(xind, yind,
                  value1, reward,
                  terminals, [:N, :W, :S, :E], π,
                  value1, value2,
                  true, true)
end

const transition = Dict(:N => (0, 1),
                        :W => (-1, 0),
                        :S => (0, -1),
                        :E => (1, 0))

trans(i::Int64, j::Int64, action::Symbol) = [i, j] + [transition[action]...]

function is_outside(world::Grid2D, i::Int64, j::Int64)
    if i < 1 || i > world.xind
        return true
    elseif j < 1 || j > world.yind
        return true
    else
        return false
    end
end

@inline get_value(world::Grid2D, i::Int64, j::Int64) = (is_outside(world, i, j)) ? -1.0 : world.value[i, j]

@inline get_reward(world::Grid2D, i::Int64, j::Int64) = world.reward_[i, j]

function Q(world::Grid2D, γ::Float64, i::Int64, j::Int64, action::Symbol, actions::Vector{Symbol})
    # if (i, j) is outside of the grid, return nothing
    q = 0.0
    
    for a in actions
        pos = trans(i, j, a)
        r = get_reward(world, i, j)
        v = get_value(world, pos...)

        cost = 0.0
        if is_outside(world, pos...)
            cost = v + r
        else
            cost = γ * v + r
        end
        if a == action
            q += 0.7 * cost
        else
            q += 0.1 * cost
        end
    end
    return q
end

function value_iteration!(world::Grid2D, γ::Float64)
    value = world.value # current value (world.value1 if world.switch == true else world.value2)
    new_value = (world.switch == true) ? world.value2 : world.value1
    π = world.π
    
    xind, yind = world.xind, world.yind
    for i in 1:xind
        for j in 1:yind
            # VI terminals only at 1st time
            if (i, j) in world.terminals
                if !world.initial
                    continue
                end
            end
            
            max_actions = Vector{Symbol}(undef, 0)
            vals = Vector{Float64}(undef, 0)
            
            for action in world.action_sets
                val = Q(world, γ, i, j, action, world.action_sets)
                push!(vals, val)
            end

            max_val = maximum(vals)
            for i in 1:length(vals)
                if abs(max_val - vals[i]) < 0.005
                    push!(max_actions, world.action_sets[i])
                end
            end

            new_value[i, j] = max_val
            π[i, j] = max_actions
        end
    end

    if world.initial
        for (i, j) in world.terminals
            world.value1[i, j] = new_value[i, j]
        end
        world.initial = false
    end
    
    if world.switch == true
        world.value = world.value2
        world.switch = false
        return norm(world.value1 - world.value2)
    else
        world.value = world.value1
        world.switch = true
        return norm(world.value1 - world.value2)
    end
end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function draw(world::Grid2D)
    xind, yind = world.xind, world.yind
    p = plot(xlims=(0.0, xind), xticks=0:1:xind, label=nothing, grid=true, grid_width=1.0, gridstyle=:solid, gridalpha=1.0, aspect_ratio=:equal)
    p = plot!(ylims=(0.0, yind), yticks=0:1:yind, grid=true, grid_width=1.0, gridstyle=:solid, gridalpha=1.0)
    min_value = Inf
    max_value = -Inf
    for i in 1:xind
        for j in 1:yind
            draw_cell!(world, i, j, p)
            value = get_value(world, i, j)
            if value > max_value
                max_value = value
            end
            if value < min_value
                min_value = value
            end
        end
    end

    for i in 1:xind
        for j in 1:yind
            value = get_value(world, i, j)
            draw_policy!(world, i, j, p)
        end
    end
    return p
end

function draw_cell!(world::Grid2D, i::Int64, j::Int64, p)
    x, y = i - 0.5, j - 0.5
    annotate!(p, x, y, text("$(round(get_value(world, i, j), sigdigits=3))", 5))
end

function draw_policy!(world::Grid2D, i::Int64, j::Int64, p)
    actions = world.π[i, j]
    x, y = i - 0.5, j - 0.5
    for action in actions
        if action == :N
            p = quiver!([x], [y], quiver=([0], [0.4]), color=:black, alpha=0.3, linewidth=3.0, arrowscale=0.03)
        elseif action == :W
            p = quiver!([x], [y], quiver=([-0.4], [0]), color=:black, alpha=0.3, linewidth=3.0, arrowscale=0.03)
        elseif action == :S
            p = quiver!([x], [y], quiver=([0], [-0.4]), color=:black, alpha=0.3, linewidth=3.0, arrowscale=0.03)
        elseif action == :E
            p = quiver!([x], [y], quiver=([0.4], [0]), color=:black, alpha=0.3, linewidth=3.0, arrowscale=0.03)
        end
    end
end

function main_cli(γ = 0.9)
    xind, yind = 10, 10
    rewards = Dict((9, 3) => 10.0, (8, 8) => 3.0, (4, 6) => -5.0, (4, 3) => -10.0)
    terminals = [(9, 3), (8, 8)]
    world = Grid2D(xind, yind, rewards, terminals)

    delta = 1.0
    while delta > 0.00001
        delta = value_iteration!(world, γ)
        println(delta)
    end
    p = draw(world)
    plot(p)
end

function main_gif(γ = 0.9)
    xind, yind = 10, 10
    rewards = Dict((9, 3) => 10.0, (8, 8) => 3.0, (4, 6) => -5.0, (4, 3) => -10.0)
    terminals = [(9, 3), (8, 8)]
    world = Grid2D(xind, yind, rewards, terminals)
    
    anim = @animate for i in 1:1
        value_iteration!(world, γ)
        p = draw(world)
    end

    gif(anim, "grid.gif", fps=2)
end
