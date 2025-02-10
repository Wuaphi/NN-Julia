using Serialization
using DataFrames
using Statistics

mutable struct NN
    in::Vector
    deepbias::Array{Tuple{Matrix{Float32}, Vector{Float32}}}
    activation::Tuple{Function,Int}
    outActivation::Tuple{Function,Int}
    lossf::Int
    actder::Tuple{Function, Int}
    oactder::Tuple{Function, Int}
end

struct Layerresult
    results::Array{Vector{Float32}}
end

function createNN(inp::Int, out::Int, deep::Int, deeplayer::Int; activation::Int=3, outActivation::Int = -1, lossf::Int = 0, normfunc::Int = -2, normfuncout::Int = -2)
    fin = fill(missing, inp)
    fdeepbias = Array{Tuple{Matrix{Float32}, Vector{Float32}}}(undef, deeplayer)
    counter = 1

    activations = Dict(
        -1 => (x -> x, nothing),
        0 => (x -> max(x, 0), (in, out) -> randn(Float32, out, in) * sqrt(2/in)),
        1 => (x -> 1/ (1+ exp(-x)), (in, out) -> randn(Float32, out, in) * sqrt(2/ (in + out))),
        2 => (x -> tanh(x), (in, out) -> randn(Float32, out, in) * sqrt(2/ (in + out))),
        3 => (x -> max(x, x*0.001), (in, out) -> randn(Float32, out, in) * sqrt(2/in)),
        4 => (x -> [exp(sui)/ sum(exp.(x)) for sui in x], nothing)
    )

    actders = Dict(
        -1 => x -> 1,
        0 => x -> x <= 0 ? 0 : 1,
        1 => x -> exp(-x)/((1+exp(-x))^2),
        2 => x -> 1 - tanh(x)^2,
        3 => x -> x <= 0 ? 0.001 : 1,
        4 => nothing
    )

    
    if activation in keys(activations)
        (f, init) = activations[activation]
        isnothing(init) && throw(ArgumentError)
    else 
        throw(ArgumentError)
    end

    if outActivation in keys(activations)
        of = activations[outActivation][1]
    else
        throw(ArgumentError)
    end

    while true
        if counter == 1
            fdeepbias[counter] =  (Float32.(init(inp, deep)), Float32.(rand(deep) .* 0.001))
        elseif counter == deeplayer 
            fdeepbias[counter] = (Float32.(init(deep, out)), Float32.(rand(out) .* 0.001))
            break
        else
            fdeepbias[counter] = (Float32.(init(deep, deep)), Float32.(rand(deep) .* 0.001))
        end
        counter +=1
    end
    return NN(fin, fdeepbias, (f,activation), (of,outActivation), lossf, (actders[activation], activation), (actders[outActivation], outActivation))
end

function saveNN(neural::NN, location::String)
    try
        open(location,"w") do f 
            serialize(f, neural)
        end
        return true
    catch
        return false
    end
end

function loadNN(location::String)
    try
        open(location,"r") do f
            return deserialize(f)
        end
    catch
        return false
    end
end

function trunNN(neural::NN, current::Vector{Float32}) 
    length(current) != length(neural.in) && throw(ArgumentError)
    activation = neural.activation[1]
    softmax = neural.activation[2] == 4
    results = [current]
    for (weights,bias) in neural.deepbias
        curr = (weights * current) + bias
        push!(results, curr)
        !(softmax) ? current = activation.(curr) : current = activation(curr)
    end
    return Layerresult(results)
end

function runNN(neural::NN, current::Vector{Float32})
    length(current) != length(neural.in) && throw(ArgumentError)
    activation = neural.activation[1]
    softmax = neural.activation[2] == 4
    for (idx,(weights, bias)) in enumerate(neural.deepbias)
        last = idx == length(neural.deepbias) 
        last && (activation = neural.outActivation[1])
        !(softmax) || last ? (current = activation.((weights * current) + bias)) : (current = activation((weights * current) + bias))
    end
    return current 
end

function clip(gradient, mins, maxs)
    return min(max(gradient, mins), maxs)
end

function backpropagateNN!(neural::NN, r::Layerresult, goal::Vector{Float32}; learnrate::Float32 = Float32(0.01), showloss::Bool = false, clips::Bool = false, clipper::Int = 5)

    save = copy(neural.deepbias)

    actder = neural.actder[1]
    neural.outActivation[2] != 4 ? oactder = neural.oactder[1] : (neural.lossf != 1 && throw(ArgumentError))
    error = nothing
    gradient = nothing
    activation = neural.activation[1]

    for realidx in reverse(1:length(save))
        (weights,biases) = save[realidx]

        if realidx == length(save)
            out = neural.outActivation[1]
            output = (out.(r.results[realidx+1]))
            neural.lossf == 1 && neural.outActivation[2] == 4 ? (error = output - goal) : (error = (output -  goal) .* oactder.(r.results[realidx+1]))
            gradient = error * transpose(activation.(r.results[realidx]))
            clips && (gradient = clip.(gradient, -clipper, clipper))
        else
            error = (transpose(save[realidx+1][1]) * error) .* actder.(r.results[realidx+1])
            gradient = error * transpose(activation.(r.results[realidx]))
            clips && (gradient = clip.(gradient, -clipper, clipper))
        end
        neural.deepbias[realidx] = (weights .- learnrate * gradient, biases .- learnrate * error)
    end
    showloss ? (return (sum(r.results[end] - goal).^2) / length(goal)) : nothing
end


function trainNN!(neural::NN, input::Vector{Vector{Float32}}, truth::Vector{Vector{Float32}}; repetitions::Int = -1, learnrate::Float32 = Float32(0.01), learngoal::Float32 = Float32(0.01), showloss::Bool = false, clips::Bool = false, clipper::Int = 5)
    length(input[1]) != size(neural.deepbias[1][1])[2] || length(truth[1]) != size(neural.deepbias[end][1])[1] ? throw(ArgumentError) : nothing
    showloss && (losses = [])

    if repetitions != -1
        while repetitions > 0
            repetitions -= 1
            for idx in eachindex(input)
                predicted =  trunNN(neural, input[idx])
                loss = backpropagateNN!(neural, predicted, truth[idx], learnrate=learnrate, showloss=showloss, clips=clips, clipper=clipper)
                showloss && push!(losses, loss)
            end
        end
    else
        losses = fill(learngoal, length(input))
        counter = 1
        while !(all(abs.(losses[length(losses) - length(input)+1 : end]) .< learngoal))
            counter += 1
            for idx in eachindex(input)
                predicted = trunNN(neural, input[idx])
                loss = backpropagateNN!(neural, predicted, truth[idx], learnrate=learnrate, showloss=showloss, clips=clips, clipper=clipper)
                push!(losses, loss)
                !(showloss) && deleteat!(losses, 1 : length(losses) - length(input))
            end
            (showloss && counter % 1000 == 0) && println("INFO: Loss at $counter's iteration -> " * string(abs.(losses[length(losses) - length(input)+1 : end])) * " Complete Loss -> " * string(sum(abs.(losses[length(losses) - length(input)+1 : end]))))
        end
    end
    showloss && (return losses)
end


function learnNN(indf::DataFrame, trainon::Vector{String}; status::Bool = true, deepneurons::Int = 16, layer::Int = 3, learngoal::Float64 = Float64(-1), activation::Int = 3, outActivation::Int = -1, lossf::Int = -1, accuracy::Float64 = 0.01, learnrate::Float64 = 0.01)
    !(all([in(x, names(indf)) for x in trainon])) && throw(ArgumentError)
    truth = collect.(eachrow(Float32.(select(df, trainon...))))
    input = [Float32.(x) for x in collect.(eachrow(select(df, Not(trainon...))))]
    flatinput = collect(Iterators.flatten(input))
    flatoutput = collect(Iterators.flatten(truth))
    if lossf == -1
        if any((flatinput .< -1) .|| (flatinput .> 1))
            lossf = 0
        else
            length(unique(flatinput)) > (length(input) * length(input[1])) / 2 ? (lossf = 0) : (lossf = 1)
        end
    end
    lossf == 0 ? (problem = "regression") : (problem = "classification")

    status && println("INFO: parsed & interpreted inputs. Interpreted as a $problem problem. If interpreted falsely, please define problem type manually.")
    neural = createNN(length(input[1]), length(truth[1]), deepneurons, layer, activation=activation, outActivation=outActivation, lossf=lossf)
    learngoal == Float64(-1) ? (lossf == 0 ? (learngoal = Float32((sum(flatoutput) / length(flatoutput)) * accuracy)) : (learngoal = Float32(0.95))) : (learngoal = Float32(learngoal))

    status && println("INFO: created neural network with $layer layers and $deepneurons neurons each. Learngoal estimated as $learngoal. Starting training now.")

    trainNN!(neural, input, truth, learnrate=Float32(learnrate), learngoal=learngoal, showloss=status)

    status && println("INFO: neural network successfully trained.")
    return neural
end

function dfrunNN(neural::NN, df::DataFrame, trainon::Vector{String}; similar::Bool = false, learngoal::Float32 = Float32(0.001))
    !(all([in(x, names(df)) for x in trainon])) && throw(ArgumentError)
    truth = collect.(eachrow(Float32.(select(df, trainon...))))
    input = [Float32.(x) for x in collect.(eachrow(select(df, Not(trainon...))))]
    results = [runNN(neural, batch) for batch in input]
    !(similar) ? (return results, truth) : (return abs.(results - truth) .< learngoal)
end
