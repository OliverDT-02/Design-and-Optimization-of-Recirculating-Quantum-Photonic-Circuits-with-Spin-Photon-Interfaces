#Load the necessary packages
using QuantumToolbox
using LinearAlgebra
using Random
using Plots
using Zygote
using ForwardDiff
using Optimization
using OptimizationOptimJL
using CairoMakie
using BenchmarkTools: @btime, @benchmark
using Printf
using OptimizationOptimisers
using QuantumControl
using QuantumPropagators: Cheby, Newton
using ExponentialAction
using Profile
using SparseArrays
using PProf
using SciMLSensitivity
using Enzyme
using FiniteDiff
using ForwardDiff
using ExponentialUtilities
using JLD2 
using DSP
#--
job_index = parse(Int, get(ENV, "LSB_JOBINDEX", "1"))

const N_fock =  3; #dimension for the Fock states
const N_rails = 2; #number of rails



#define the identity and anhilitation operators using the QuantumToolbox package 
#I define other operators as well, also the states

#operators for the cavity
c  = destroy(N_fock);
id_c = qeye(N_fock);

#operators and states for the lambda system 
g_1  = basis(3, 0); #consider that [1, 0, 0] is the |g₁⟩  state
g_2 = basis(3, 1); #consider that [0, 1, 0] is the |g₂⟩  state 
e = basis(3, 2); #consider that [0, 0, 1] is the |e⟩  state

σ₊ = e*g_1'; # |e⟩⟨g_1| 
T₊ = g_2*g_1'; # |g_2⟩⟨g_1|
id_Λ = qeye(3); #\

#tensor the Fock representation with the Lambda system representation for a single state. Basically a unit of state will be |state⟩ = |Fock⟩ ⊗ |Λ⟩ 
c_state = tensor(c, id_Λ);
id_state = tensor(id_c, id_Λ);
σ₊_state = tensor(id_c, σ₊);
T₊_state = tensor(id_c, T₊);


#Now tensor for multiple states (so far just two systems). Now I extend to |state⟩ ⊗ |state⟩
c_1 = tensor(c_state, id_state); #This is then equivalent to ̂c ⊗ I_Λ ⊗ (I_c ⊗ I)
c_2 = tensor(id_state, c_state); 

σ₊_1 = tensor(σ₊_state, id_state);
σ₊_2 = tensor(id_state, σ₊_state);

T₊_1 = tensor(T₊_state, id_state);
T₊_2 = tensor(id_state, T₊_state);

#I need this list format for the construction of the Hamiltonian 
c_ops = [c_1, c_2]; #I will also use this one for expectation values 
σ₊_ops = [σ₊_1, σ₊_2];
T₊_ops = [T₊_1, T₊_2];
c_σ₊_ops = [c_1*σ₊_1, c_2*σ₊_2];


#define operators that will tell us the populations of different elements of the system, so basically build projectors
g1_projector_state = tensor(id_c, g_1*g_1'); #basically I_c ⊗ |g1⟩⟨g1|
g2_projector_state = tensor(id_c, g_2*g_2');
e_projector_state = tensor(id_c, e*e');

#make it a vector of operators for each system
g1_projector_ops = [tensor(g1_projector_state, id_state), tensor(id_state, g1_projector_state)]; #first element is the projector for |g1)(g1| for state 1, second is the same projector for state 2
g2_projector_ops = [tensor(g2_projector_state, id_state), tensor(id_state, g2_projector_state)];
e_projector_ops = [tensor(e_projector_state, id_state), tensor(id_state, e_projector_state)];

#new operators for expectation values
n1Fock = [tensor(basis(N_fock, 1)*basis(N_fock, 1)', id_Λ,  id_state), tensor(id_state, basis(N_fock, 1)*basis(N_fock, 1)', id_Λ)]; #(|1⟩⟨1| ⊗ I_λ) ⊗ I_s  and I_s ⊗ (|1⟩⟨1| ⊗ I_λ)
n2Fock = [tensor(basis(N_fock, 2)*basis(N_fock, 2)', id_Λ,  id_state), tensor(id_state, basis(N_fock, 2)*basis(N_fock, 2)', id_Λ)];

#I use this to have a version of the operators that are NOT Quantum Objects (QuantumToolbox)
ops = (c_ops = [ComplexF64.(sparse(c.data)) for c in c_ops],
    g1_projector_ops = [ComplexF64.(sparse(g1_projector.data)) for g1_projector in g1_projector_ops],
    g2_projector_ops = [ComplexF64.(sparse(g2_projector.data)) for g2_projector in g2_projector_ops],
    e_projector_ops = [ComplexF64.(sparse(e_projector.data)) for e_projector in e_projector_ops],
    T₊_ops = [ComplexF64.(sparse(T₊.data)) for T₊ in T₊_ops],
    σ₊_ops = [ComplexF64.(sparse(σ₊.data)) for σ₊ in σ₊_ops],
    c_σ₊_ops = [ComplexF64.(sparse(c_σ₊.data)) for c_σ₊ in c_σ₊_ops]
    );  
## optimize function
    function make_opt_problem(cost, u0, params, ad_backend=Optimization.AutoZygote())
    f_obj(u, p) = cost(u, p)


    optf = OptimizationFunction(f_obj, Optimization.AutoZygote())

    if isnothing(ad_backend)
        optf = OptimizationFunction(cost) # No AD
    else
        optf = OptimizationFunction(cost, ad_backend)
    end

    #optf = OptimizationFunction(f_obj, Optimization.AutoForwardDiff())
    #optf = OptimizationFunction(f_obj, Optimization.AutoEnzyme())
  
    return OptimizationProblem(optf, u0, params) #lb = lower_bounds, ub = upper_bounds )
end

## Evolution
function create_H_terms(κ, C, g=1, Δ_c=0, Δ_1=0, Δ_2=0)
    H0 = sum(sqrt(κ[i]' * κ[j] + 1e-12) * C[i, j] * c_ops[i]' * c_ops[j] 
             for i in 1:N_rails for j in 1:N_rails)  + sum(Δ_c*c_ops[i]'*c_ops[i] + Δ_2*g2_projector_ops[i] + Δ_1*e_projector_ops[i] + g*(c_σ₊_ops[i] + c_σ₊_ops[i]') for i in 1:N_rails) #time independent

#This is the matrix that multiplies the term related with the spin transition
    H_Ω = map(1:N_rails) do i #for this I need to consider that I have something like Ω|g2)(g1| + Ω^*|g1)(g2| but Ω is a complex number, and remember that here I need to work with reals. So I need to rexpress everything like 
                                #Ω.re(|g2)(g1| + |g1)(g2|) + iΩ.im(|g2)(g1| - |g1)(g2|)
    H_re = T₊_ops[i] + T₊_ops[i]'
    H_im = 1im*(T₊_ops[i] - T₊_ops[i]')

    [H_re, H_im]
    end

#This is the matrix that multiplies the term related with the optical transition
    H_ϵ = map(1:N_rails) do i
    H_re = σ₊_ops[i] + σ₊_ops[i]'
    H_im = 1im*(σ₊_ops[i] - σ₊_ops[i]')

    [H_re, H_im]
    end

    H0_QO = H0
    H_Ω_QO = H_Ω
    H_ϵ_QO = H_ϵ

    H0_QO = sparse(H0_QO)
    H_Ω_QO = [[sparse(H_real_imag) for H_real_imag in H_rail] for H_rail in H_Ω_QO];
    H_ϵ_QO = [[sparse(H_real_imag) for H_real_imag in H_rail] for H_rail in H_ϵ_QO];

    H0 = ComplexF64.(sparse(H0.data))
    H_Ω = [[ComplexF64.(sparse(H_real_imag.data)) for H_real_imag in H_rail] for H_rail in H_Ω];
    H_ϵ = [[ComplexF64.(sparse(H_real_imag.data)) for H_real_imag in H_rail] for H_rail in H_ϵ];

    return(H0 = H0, H_Ω = H_Ω, H_ϵ = H_ϵ, H0_QO = H0_QO, H_Ω_QO = H_Ω_QO, H_ϵ_QO = H_ϵ_QO)
end


function create_H(H0, H_Ω, H_ϵ, Ω, ϵ) #now the matrices are constants and I do not need to recompute them every time I call the function

    H = H0 + sum(H_Ω[i][1]*real(Ω[i]) + H_Ω[i][2]*imag(Ω[i]) + H_ϵ[i][1]*real(ϵ[i]) + H_ϵ[i][2]*imag(ϵ[i]) for i in 1:N_rails) 
end

#Evolutions
function Hamiltonian_evolution(ψ0::AbstractVector{<:Complex}, H0, H_Ω::AbstractVector, H_ϵ::AbstractVector, Ω, ϵ, τ; t_wait = nothing)
    """ 
    Computes the evolution, I create the Hamiltonian in the most efficient way using pre-computed matrices.
    I compute the evolution using the function "ExponentialAction.expv" which should be very fast. This function is special because it works with Zygote.
    """
    ψ = ψ0
    
    for n in 1:n_seg
        #Build the Hamiltonian for each time segment 
        H = create_H(H0, H_Ω, H_ϵ, view(Ω, n, :), view(ϵ, n, :))
        ψ = ExponentialAction.expv(-im*τ[n], H, ψ) 
        
    end
    
    #part where there is NO control fields
    if t_wait != nothing  &&  t_wait > 0
        H = create_H(H0, H_Ω, H_ϵ, zeros(size(Ω, 2)), zeros(size(ϵ, 2)))
        ψ = ExponentialAction.expv(-im*t_wait, H, ψ)
    end

    return ψ
end

##Compute partial traces
function partial_trace_cavities(ψ)
    #obtain a clear index representation of the state
    #The following two lines are STRANGE, BUT THEY WORK
    ψ_clear = reshape(ψ, 3, N_fock, 3, N_fock) #now I can easily access the coefficient of an element like |2) ⊗ |2) ⊗ |2) ⊗ |2) (where now I use 2 also to represent the matter state (|g2)) #notice that the order is different but this is because that is how reshape works
    ψ_clear_permuted = permutedims(ψ_clear, (4,3,2,1)) #the reshape messes up the ordering, without this everything breaks down
    # Compute all elements using array comprehension (non-mutating)
    ρ_reduced = [sum(ψ_clear_permuted[n1, m1, n2, m2] * conj(ψ_clear_permuted[n1, m1p, n2, m2p]) 
                     for n1 in 1:N_fock, n2 in 1:N_fock)
                 for m1 in 1:3, m2 in 1:3, m1p in 1:3, m2p in 1:3]

    ρ_reduced = permutedims(ρ_reduced, (2,1,4,3))
    return reshape(ρ_reduced, 9, 9)
end

##
function create_ΩMatrix(p::AbstractVector{<:Real}, A::Real, pulse)
    #Ω is a vector containing the matrix components of all the parameters for the Ω terms (all the rails and all the segments)
    #A defines the amplitude of this reparameterization
    #pulse is needed to modulate the signal in the extremes 
    σ(x) = A*(1/(1 + exp(-x))-0.5)
    len_p = length(p)
    real_Ω = reshape(view(p, 1:len_p÷2), n_seg, N_rails) #I use ÷ because this is a truncated integer division (rather than / that returns a float and causes instability type issues)
    im_Ω = reshape(view(p, len_p÷2+1:lastindex(p)), n_seg, N_rails)

    Ω_Matrix = σ.(real_Ω) + 1im*σ.(im_Ω)  #Do this to restrict more naturally the domain, Now the domain is (-A/2, A/2)
    Ω_Matrix = Ω_Matrix .* pulse #Consider that we want the signal to be 0 in the extremes (at the beginning and at the end)

    return Ω_Matrix
end



#for a given input parameters it outputs the actual ϵ values for the system
function create_ϵMatrix(p::AbstractVector{<:Real}, B::Real, pulse)
    σ(x) = B*(1/(1 + exp(-x))- 0.5) 
 
    len_p = length(p)
    real_ϵ = reshape(view(p, 1:len_p÷2), n_seg, N_rails)
    im_ϵ = reshape(view(p, len_p ÷ 2+1:lastindex(p)), n_seg, N_rails)

    ϵ_Matrix = σ.(real_ϵ) + 1im*σ.(im_ϵ) #Do this to restrict more naturally the domain, Now the domain is (-B/2, B/2)
    ϵ_Matrix = ϵ_Matrix .* pulse #Consider that we want the signal to be 0 in the extremes (at the beginning and at the end)

    return ϵ_Matrix
end

#this function creates a vector τ that determines the duration of each segment 
#this is parametrized such that there is a domain of the time that can be choosen. The domain of the time duration per segment is (τmin, τmax)
function create_τvector(τ, τmin::Real, τmax::Real, dt::Real)
    σ(x) = τmin + (τmax - τmin)/(1 + exp(-x)) 
    #return ceil.(σ.(τ.- 1)./dt).*dt #I use ceil and multiply and divide by dt because I want a duration that actually belongs to my time grid  
    return ceil.(σ.(τ)./dt).*dt
end

##Definitions of constants
const dt = 0.002;
const n_seg = 50; #number of segments
const τmin = dt; #minimum duration of a single segment
const τmax = 100*dt; #total max time of τmax*n_seg = 40  (for now)
const t_wait = 2.0; 

#define constants from the Hamiltonian
const Δ_2 = 0; #detunings 
const Δ_1 = 0;
const Δ_c = 0;
const Ω_amplitude = 50; #amplitudes for the control signals. This means it is NOT the max value. The max value is this divided by two. 
const ϵ_amplitude = 50;
const g = 1; #coupling between matter and light

κ_value = parse(Float64, ARGS[1])
κ = [κ_value, κ_value] #coupling between the cavity and the waveguides
C = [0 1; 1 0] #Matrix that appears from the SLH
#τvector = 0.5*ones(n_seg)
#const  pulse = tukey_window(n_seg);
const pulse = tukey(n_seg, 0.3);
#const cumsum_τvector = cumsum(τvector)

H_terms = create_H_terms(κ, C)

H0 = H_terms.H0
H_Ω = H_terms.H_Ω
H_ϵ = H_terms.H_ϵ

H0_QO = H_terms.H0_QO
H_Ω_QO = H_terms.H_Ω_QO
H_ϵ_QO = H_terms.H_ϵ_QO

#creation of initial state
ψ0 = tensor(basis(N_fock, 0), g_1, basis(N_fock, 0), g_1); #[|0) ⊗ |g_1)] ⊗ [|0) ⊗ |g_1)]
ψ0_QO = ψ0
ψ0 = ψ0.data; #we just want the vector representation
#desired state (notice that it is just defined in the matter system)
ψT_matter = (tensor(g_1, g_1) + tensor(g_2, g_2))/sqrt(2);
ψT_matter = ψT_matter.data;

function cost_this(u, params)
    (; Ω_amplitude, ϵ_amplitude, ψ0, t_wait, ψT_matter, κ, H0, H_Ω, H_ϵ, tF) = params
  
    #define the indexes for the functions
    idx_Ω = 1:2*n_seg*N_rails;
    idx_ϵ = 2*n_seg*N_rails+1 : 4*n_seg*N_rails;


    #define the things that will actually go into the Hamiltonian (these depend on the parameter p)
    ΩMatrix = create_ΩMatrix(view(u, idx_Ω), Ω_amplitude, pulse);
    ϵMatrix = create_ϵMatrix(view(u, idx_ϵ), ϵ_amplitude, pulse);
    τvector = tF/n_seg*ones(n_seg)



    ψf = Hamiltonian_evolution(ψ0, H0, H_Ω, H_ϵ, ΩMatrix, ϵMatrix, τvector; t_wait = t_wait)

    ρm = partial_trace_cavities(ψf)
    Fsum = real(ψT_matter'*ρm*ψT_matter)

    return (1-Fsum)
end

##Code for the specific task of interest 
function solve_problem_time(cost, u0, params; adam_iters=100, lbfgs_iters=100, adam_lr = 0.02, target_cost=5e-3)

    # definition of a callback function
    callback = function (state, loss)
        if state.iter % 10 == 0
            println("Iteration: $(state.iter) | Cost: $(round(loss, digits=6))")
        end
        
        # Halt optimization if loss drops below the threshold
        if loss < target_cost
            println("Stopping early at iteration $(state.iter): Cost $(round(loss, digits=6)) is below threshold $(target_cost).")
            return true 
        end
        
        return false 
    end
    
    if adam_iters == 0
        prob = make_opt_problem(cost, u0, params)
        println("Starting L-BFGS phase (only)...")
        sol_final = Optimization.solve(prob, OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters, callback = callback)
        println("L-BFGS finished with final cost: ", sol_final.objective)
        return sol_final
    else
        prob = make_opt_problem(cost, u0, params)
        println("Starting Adam phase...")
        sol_adam = Optimization.solve(prob, OptimizationOptimisers.Adam(adam_lr); maxiters = adam_iters, callback = callback)
        println("Adam finished with cost: ", sol_adam.objective)
        
        # If no L-BFGS iters were requested, OR if Adam already hit the target cost, return early
        if lbfgs_iters == 0 || sol_adam.objective < target_cost
            return sol_adam
        else
            prob_lbfgs = remake(prob, u0 = sol_adam.u)
            println("Starting L-BFGS phase...")
            sol_final = Optimization.solve(prob_lbfgs, OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters, callback = callback)
            println("L-BFGS finished with final cost: ", sol_final.objective)
            return sol_final
        end
    end
end


# A helper function to handle the inner two loops
function kappa_time(κ, t_list)
    # Sort b_list ascending so the first match is guaranteed to be the smallest
    for t in t_list 
        succes_for_this_t = false

        for _ in 1:20
            # Assuming your optimization function takes a, b, and the guess
            params = (
            Ω_amplitude = Ω_amplitude, 
            ϵ_amplitude = ϵ_amplitude,  
            ψ0 = ψ0, 
            t_wait = t_wait, 
            ψT_matter = ψT_matter,
            κ = κ,
            H0 = H0,
            H_Ω = H_Ω,
            H_ϵ =  H_ϵ,
            tF = t
    #τvector = τvector
        );
            u_random = 0.5*(rand(4*n_seg*N_rails) .- 0.5);
            println("Time $t")
            sol = solve_problem_time(cost_this, u_random, params, lbfgs_iters=500);
            
            if sol.objective <= 5e-3
                succes_for_this_t = true
                break
            end
        end
            
            if !succes_for_this_t
                return t  # Instantly exits both loops and returns this 'b'
            end
        end

    return NaN # Return NaN (or nothing) if no 'b' satisfies the problem
end

#--------------------
t_range = 5:-0.1:2
t_value = kappa_time(κ_value, t_range)

# 1. Initialize an array to store the results. 

save_dir = "random_iterations" #define the folder
mkpath(save_dir) 

file_name = "kappa_time_value=$(κ[1])_$(job_index).csv"
full_path = joinpath(save_dir, file_name)

    open(full_path, "w") do file
    #
        println(file, "$job_index, $κ_value, $t_value")
    end