##Essential functions used for other scripts and/or notebooks

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

#--
const N_fock =  4; #dimension for the Fock states
const N_rails = 2; #number of rails



#define the identity and anhilitation operators using the QuantumToolbox package 
#define basis states

#operators for the cavity
c  = destroy(N_fock); #anhilitation operator
id_c = qeye(N_fock); #identity operator

#operators and states for the lambda system 
g_1  = basis(3, 0); #consider that [1, 0, 0] is the |g₁⟩  state
g_2 = basis(3, 1); #consider that [0, 1, 0] is the |g₂⟩  state 
e = basis(3, 2); #consider that [0, 0, 1] is the |e⟩  state

#Other relevant operators
σ₊ = e*g_1'; # |e⟩⟨g_1| 
T₊ = g_2*g_1'; # |g_2⟩⟨g_1|
id_Λ = qeye(3); #identity operator for the Lambda system

#tensor the Fock representation with the Lambda system representation for a single state. Basically a unit of state will be |state⟩ = |Fock⟩ ⊗ |Λ⟩ 
c_state = tensor(c, id_Λ); #anhilitation operator (cavity)
id_state = tensor(id_c, id_Λ); #identity operator
σ₊_state = tensor(id_c, σ₊); # I_c ⊗ |e⟩⟨g_1| 
T₊_state = tensor(id_c, T₊); #I_c ⊗ |g_2⟩⟨g_1| 
 #all of these are operators defined for a single state


#Now tensor for multiple states (so far just two systems). Now I extend to |state⟩ ⊗ |state⟩
c_1 = tensor(c_state, id_state); #This is then equivalent to ̂c ⊗ I_Λ ⊗ (I_c ⊗ I_Λ)
c_2 = tensor(id_state, c_state);  #anhilitation operator in the 2nd cavity

σ₊_1 = tensor(σ₊_state, id_state); #σ₊ operator in the 1st Lambda system
σ₊_2 = tensor(id_state, σ₊_state); #σ₊ operator in the 2nd Lambda system

T₊_1 = tensor(T₊_state, id_state); #T₊ operator in the 1st Lambda system
T₊_2 = tensor(id_state, T₊_state); #T₊ operator in the 2nd Lambda system


#I need this list format for the construction of the Hamiltonian and for the computation of the expectation values
c_ops = [c_1, c_2]; 
σ₊_ops = [σ₊_1, σ₊_2];
T₊_ops = [T₊_1, T₊_2];
c_σ₊_ops = [c_1*σ₊_1, c_2*σ₊_2];



#define operators that will tell us the populations of different elements of the system, so basically build projectors (for a single system)
g1_projector_state = tensor(id_c, g_1*g_1'); #basically I_c ⊗ |g1⟩⟨g1|
g2_projector_state = tensor(id_c, g_2*g_2');
e_projector_state = tensor(id_c, e*e');

#make it a vector of operators for each system
g1_projector_ops = [tensor(g1_projector_state, id_state), tensor(id_state, g1_projector_state)]; #first element is the projector for |g1)(g1| for state 1, second is the same projector for state 2
g2_projector_ops = [tensor(g2_projector_state, id_state), tensor(id_state, g2_projector_state)];
e_projector_ops = [tensor(e_projector_state, id_state), tensor(id_state, e_projector_state)];

#new operators for expectation values, again it is a vector.
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

##
#use this for natural evolution of the control signal (0 at the extremes)
function tukey_window(N::Int; α=0.3)
    w = ones(Float64, N)
    edge = floor(Int, α*(N-1)/2)

    for n in 0:N-1
        k = n + 1
        if n < edge
            w[k] = 0.5 * (1 - cos(π * n / edge))
        elseif n > (N-1-edge)
            w[k] = 0.5 * (1 - cos(π * (N-1-n) / edge))
        end
    end
    return w
end
#
##
#Optimization task

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
#I make this a function because I can later reuse the code, and it is just like a template I need to follow


function solve_problem(cost, u0, params; adam_iters=100, lbfgs_iters=100, adam_lr = 0.02)

    #definition of a callback function
    callback = function (state, loss)
        if state.iter % 10 == 0

            println("Iteration: $(state.iter) | Cost: $(round(loss, digits=6))")
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
        if lbfgs_iters == 0
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

#Matrices are actually constants so I can make this more efficient, by creating all the constat matrix terms from the beginning.

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

#Create constant terms for the Liouvillian
function create_terms_L(collapse_ops)
    spre(A, I) = kron(I, sparse(A))
    spost(B, I) = kron(sparse(transpose(B)), I)
    sprepost(A, B) = kron(sparse(transpose(B)), sparse(A))


    N = size(H0, 1)
    I_cache = sparse(I, N, N)

    L_H(H) = -1im * (spre(H, I_cache) - spost(H, I_cache))
    #H0 term when I convert to Liouvillian
    L_H0 = L_H(H0)

    #I do the corresponding equivalents for the Liouvillian formalism
    L_HΩ = map(1:N_rails) do i
        L_re = L_H(H_Ω[i][1])
        L_im = L_H(H_Ω[i][2])

        [L_re, L_im]
    end

    L_Hϵ = map(1:N_rails) do i
        L_re = L_H(H_ϵ[i][1])
        L_im = L_H(H_ϵ[i][2])

        [L_re, L_im]
    end

    L_diss = sum( sparse(sprepost(c, c')) - 0.5*(spre(c'*c, I_cache) + spost(c'*c, I_cache)) for c in collapse_ops)

    return (L_H0 = L_H0, L_HΩ = L_HΩ,  L_Hϵ = L_Hϵ, L_diss = L_diss)
end


function create_Liouvillian(L_H0, L_HΩ, L_Hϵ, L_diss, Ω, ϵ)
   L = L_H0 + L_diss + sum(L_HΩ[i][1]*real(Ω[i]) + L_HΩ[i][2]*imag(Ω[i]) + L_Hϵ[i][1]*real(ϵ[i]) + L_Hϵ[i][2]*imag(ϵ[i]) for i in 1:N_rails)
end


##Evolutions (Dynamics)
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

#Evolution with loss
function Liouvillian_evolution(ψ0, L_H0, L_HΩ, L_Hϵ, L_diss, Ω, ϵ,  τ; t_wait = nothing)
    N = length(ψ0)
    ρ_vec = vec(ψ0*ψ0') #I consider the density matrix and I vectorize it

    for n in 1:n_seg
        L = create_Liouvillian(L_H0, L_HΩ, L_Hϵ, L_diss, view(Ω, n, :), view(ϵ, n, :))
        ρ_vec = ExponentialAction.expv(τ[n], L, ρ_vec) #I propagate with exponential as before, but without the 1im
    end

    if t_wait != nothing  &&  t_wait > 0
        L = create_Liouvillian(L_H0, L_HΩ, L_Hϵ, L_diss, zeros(size(Ω, 2)), zeros(size(ϵ, 2)))
        ρ_vec = ExponentialAction.expv(t_wait, L, ρ_vec)
    end

    ρ = reshape(ρ_vec, N, N) #return density matrix (no vectorized)
end

#Evolution with loss (using mesolve)
function Liouvillian_evolution_mesolve(ψ0, H0, H_Ω, H_ϵ, Ω, ϵ, collapse_ops, τ, t_wait)
  
    ρ = Qobj(ψ0*ψ0', dims = (N_fock, 3, N_fock, 3)) #I need to convert my density matrix in a QuantumObject with its specified dimensions

    for n in 1:n_seg
        H = create_H(H0, H_Ω, H_ϵ, view(Ω, n, :), view(ϵ, n, :)) #First I create my Hamiltonian 
        H = Qobj(H, dims = (N_fock, 3, N_fock, 3)) #then I make it a QuantumObject with its specified dimensions 
        ρ = mesolve(H, ρ, 0:dt:τ[n], collapse_ops, progress_bar = Val(false), saveat = τ[n]).states[end] #I call the solver for master equation mesolve
    end

    if t_wait != nothing  &&  t_wait > 0
        H = create_H(H0, H_Ω, H_ϵ, zeros(size(Ω, 2)), zeros(size(Ω, 2)))
        H = Qobj(H, dims = (N_fock, 3, N_fock, 3))
        ρ = mesolve(H, ρ, 0:dt:t_wait, collapse_ops, progress_bar = Val(false), saveat = t_wait).states[end] #notice how I just compute from the beginning to the end. This actually makes no difference in the quality of the results
    end
    ρ
end


## partial traces (for pure states and mixed states)
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

function partial_trace_cavities_density(ρ)
    ρ_raw = reshape(ρ, 3, N_fock, 3, N_fock, 3, N_fock, 3, N_fock) #convert the density matrix into a tensor with easy access to the indices 
    ρ_permuted = permutedims(ρ_raw, (4, 3, 2, 1, 8, 7, 6, 5))  #the ordering of the tensor product is different from the one of the reshape function, that is why I need to do this
   
    ρ_reduced = [sum(ρ_permuted[n1, m1, n2, m2, n1, m1p, n2, m2p] for n1 in 1:N_fock, n2 in 1:N_fock)
        for m1 in 1:3, m2 in 1:3, m1p in 1:3, m2p in 1:3]
    ρ_reduced = permutedims(ρ_reduced, (2,1,4,3))
    return reshape(ρ_reduced, 9, 9)
end

##Creation of the control fields given the parameters 
#for a given input parameters it outputs the actual (Ω values for the system)
#because remember than the parameters are different from the actual values that we are inputting to the system
#the outputs of the functions is what we put as an input to the Hamiltonian 

function create_ΩMatrix(p::AbstractVector{<:Real}, A::Real, pulse)
    #p is a vector containing the matrix components of all the parameters for the Ω terms (all the rails and all the segments)
    #A defines the amplitude of this reparameterization
    #pulse is needed to modulate the signal in the extremes 
    σ(x) = A*(1/(1 + exp(-x))-0.5) #reparameterization function
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

## Cost functions 
#this considers Schrodinger evolution 
function cost(u, params)

    (; Ω_amplitude, ϵ_amplitude, ψ0, t_wait, ψT_matter, κ, H0, H_Ω, H_ϵ) = params
  
    #define the indexes for the functions
    idx_Ω = 1:2*n_seg*N_rails;
    idx_ϵ = 2*n_seg*N_rails+1 : 4*n_seg*N_rails;
    idx_τ = 4*n_seg*N_rails+1 : length(u);

    #define the things that will actually go into the Hamiltonian (these depend on the parameter p)
    ΩMatrix = create_ΩMatrix(view(u, idx_Ω), Ω_amplitude, pulse);
    ϵMatrix = create_ϵMatrix(view(u, idx_ϵ), ϵ_amplitude, pulse);
    τvector = create_τvector(view(u, idx_τ), τmin, τmax, dt)

    ψf = Hamiltonian_evolution(ψ0, H0, H_Ω, H_ϵ, ΩMatrix, ϵMatrix, τvector; t_wait = t_wait)

    ρm = partial_trace_cavities(ψf)
    Fsum = real(ψT_matter'*ρm*ψT_matter)

    return 1-Fsum
end

#This considers a Master equation approach
function cost_master(u, params)
    (; Ω_amplitude, ϵ_amplitude, ψ0, t_wait, ψT_matter, κ, L_H0, L_HΩ, L_Hϵ, L_diss) = params
    #define the indexes for the functions
    idx_Ω = 1:2*n_seg*N_rails;
    idx_ϵ = 2*n_seg*N_rails+1 : 4*n_seg*N_rails;
    idx_τ = 4*n_seg*N_rails+1 : length(u);
    #define the things that will actually go into the Hamiltonian (these depend on the parameter p)
    
    ΩMatrix = create_ΩMatrix(view(u, idx_Ω), Ω_amplitude, pulse);
    ϵMatrix = create_ϵMatrix(view(u, idx_ϵ), ϵ_amplitude, pulse);
    #CMatrix = create_C_matrix([0, 1, 1, 0]);
    τvector = create_τvector(view(u, idx_τ), τmin, τmax, dt)

    ρf = Liouvillian_evolution(ψ0, L_H0, L_HΩ, L_Hϵ, L_diss, ΩMatrix, ϵMatrix, τvector; t_wait = t_wait)

    ρm = partial_trace_cavities_density(ρf)
    Fsum = real(ψT_matter'*ρm*ψT_matter)

    return 1-Fsum
end

##
#given a trajectory of a quantum state and an operator it returns the expected value throughout time
function expect_traj(ψtraj, A, isDens)
    

    if isDens
        ex = Vector{ComplexF64}(undef, length(ψtraj))
         for i in 1:length(ψtraj)
            ρ = ψtraj[i]
            ex[i] = dot(ρ', A)
         end
    else
        ex = Vector{ComplexF64}(undef, length(ψtraj))
        for i in 1:length(ψtraj)
            ψ = ψtraj[i]
            ex[i] = ψ' * A * ψ   # <ψ|A|ψ>
        end
    end
    return ex
end

#Function that computes most of the expectation values relevant
function RelevantExpectationValues(traj, tlist, ops, isDens)
    (; c_ops, g1_projector_ops, g2_projector_ops, e_projector_ops, σ₊_ops) = ops
    exp_n_rail_cavity = Array{Float64}(undef, N_rails, length(tlist));
    exp_rail_g1 = Array{Float64}(undef, N_rails, length(tlist));
    exp_rail_g2 = Array{Float64}(undef, N_rails, length(tlist)); 
    exp_rail_e = Array{Float64}(undef, N_rails, length(tlist));

    exp_n1_Fock = Array{Float64}(undef, N_rails, length(tlist));
    exp_n2_Fock = Array{Float64}(undef, N_rails, length(tlist));
    exp_c_n_Total = Array{Float64}(undef, length(tlist));
    exp_exc_n_Total = Array{Float64}(undef, length(tlist));


    function von_neumann_entropy(m)
        #specialized solvers
        λ = eigvals(Hermitian(m))
    
        #Filter out eigenvalues that are zero or slightly negative 
        safe_λ = λ[λ .> 1e-14]
    
        #Compute the Shannon entropy of the remaining eigenvalues
        return -sum(safe_λ .* log2.(safe_λ))
    end

  
    #obtain expectation values
    for i in 1:N_rails
        exp_n_rail_cavity[i, :] .= real.(expect_traj(traj, (c_ops[i]' * c_ops[i]), isDens));
        
        exp_rail_g1[i, :] .= real.(expect_traj(traj, g1_projector_ops[i], isDens));
        exp_rail_g2[i, :] .= real.(expect_traj(traj, g2_projector_ops[i], isDens));
        exp_rail_e[i, :] .= real.(expect_traj(traj, e_projector_ops[i], isDens));

        exp_n1_Fock[i, :] .= real.(expect_traj(traj, n1Fock[i].data, isDens));
        exp_n2_Fock[i, :] .= real.(expect_traj(traj, n2Fock[i].data, isDens));
       
    end
exp_c_n_Total .= real.(expect_traj(traj, (sum(c_ops[i]' * c_ops[i] for i in 1:N_rails)), isDens));
exp_exc_n_Total .= real.(expect_traj(traj, (sum(c_ops[i]' * c_ops[i] + e_projector_ops[i] for i in 1:N_rails)), isDens));

#computation of fidelity throughout time
if isDens
    fid_time = Array{Float64}(undef, length(tlist));
    purity_reduced_state = Array{Float64}(undef, length(tlist));
    ρ_squared = Array{Matrix{ComplexF64}}(undef, length(tlist));

    for (i, ρ_i) in enumerate(traj)
        ρ_reduced = ptrace(Qobj(ρ_i, dims = (N_fock, 3, N_fock, 3)), (2, 4))
        fid_time[i] = real.(ψT_matter' * ρ_reduced.data * ψT_matter);
        purity_reduced_state[i] = real.(tr(ρ_reduced.data*ρ_reduced.data));
        ρ_squared[i] =  ρ_i*ρ_i;
        vn_entropy[i] = von_neumann_entropy(ρ_reduced.data)
    end
else
    ρ_traj_traced = partial_trace_cavities.(traj);
    fid_time = real.([ψT_matter' * ρ_i * ψT_matter for ρ_i in ρ_traj_traced]);
    purity_reduced_state = real.([tr(ρ_i*ρ_i) for ρ_i in ρ_traj_traced]);
    ρ_squared = [(ψ*ψ')*(ψ*ψ') for ψ in traj];
    vn_entropy = [von_neumann_entropy(ρ_reduced) for ρ_reduced in ρ_traj_traced]
end


Tr_ρ_square = real.([tr(Qobj(ρ2, dims = (N_fock, 3, N_fock, 3))) for ρ2 in ρ_squared]);

return (; exp_n_rail_cavity, exp_rail_g1, exp_rail_g2, exp_rail_e, exp_n1_Fock, exp_n2_Fock, exp_c_n_Total, exp_exc_n_Total, fid_time, purity_reduced_state, Tr_ρ_square, vn_entropy )   #WORKING HERE 


end

#trajectory of the state using Hamiltonian dynamics
function traj_Hamiltonian(ψ0, H0, H_Ω, H_ϵ, Ω, ϵ, τ; t_wait = nothing)
    #build Hamiltonians per segment
    ψ = Qobj(ψ0, dims = (N_fock, 3, N_fock, 3))
    trajectories = [ψ]
   
    for n in 1:n_seg
        H = create_H(H0, H_Ω, H_ϵ, view(Ω, n, :), view(ϵ, n, :))
        H = Qobj(H, dims = (N_fock, 3, N_fock, 3))
        traj = sesolve(H, ψ, 0:dt:τ[n], progress_bar = Val(false), saveat=0:dt:τ[n]).states #solving the differential equation
        ψ = traj[end]
        append!(trajectories, traj[2:end]) #I need to use the second element becuase otherwise there is a mismatch with the time index
    end
     if t_wait != nothing  &&  t_wait > 0
        H = create_H(H0, H_Ω, H_ϵ, zeros(size(Ω, 2)), zeros(size(ϵ, 2)))
        H = Qobj(H, dims = (N_fock, 3, N_fock, 3))
        traj  = sesolve(H, ψ, 0:dt:t_wait, progress_bar = Val(false), saveat=0:dt:t_wait).states
        ψ = traj[end]
        append!(trajectories, traj[2:end])
    end
    return trajectories
end

#trajectory of the state using Liouvillian dynamics
function traj_Liouvillian(ψ0, H0, H_Ω, H_ϵ, Ω, ϵ, collapse_ops, τ; t_wait = nothing)
    ρ = Qobj(ψ0*ψ0', dims = (N_fock, 3, N_fock, 3))
    trajectories = [ρ]

    for n in 1:n_seg
        H = create_H(H0, H_Ω, H_ϵ, view(Ω, n, :), view(ϵ, n, :))
        H = Qobj(H, dims = (N_fock, 3, N_fock, 3))
        traj = mesolve(H, ρ, 0:dt:τ[n], collapse_ops, progress_bar = Val(false), saveat=0:dt:τ[n]).states
        ρ = traj[end]
        append!(trajectories, traj[2:end])
    end

    if t_wait != nothing  &&  t_wait > 0
        H = create_H(H0, H_Ω, H_ϵ, zeros(size(Ω, 2)), zeros(size(Ω, 2)))
        H = Qobj(H, dims = (N_fock, 3, N_fock, 3))
        traj = mesolve(H, ρ, 0:dt:t_wait, collapse_ops, progress_bar = Val(false), saveat=0:dt:t_wait).states
        ρ = traj[end]
        append!(trajectories, traj[2:end])
    end
    trajectories
end


 ops_pure = (c_ops = [ComplexF64.(sparse(c.data)) for c in c_ops],
    g1_projector_ops = [ComplexF64.(sparse(g1_projector.data)) for g1_projector in g1_projector_ops],
    g2_projector_ops = [ComplexF64.(sparse(g2_projector.data)) for g2_projector in g2_projector_ops],
    e_projector_ops = [ComplexF64.(sparse(e_projector.data)) for e_projector in e_projector_ops],
    T₊_ops = [ComplexF64.(sparse(T₊.data)) for T₊ in T₊_ops],
    σ₊_ops = [ComplexF64.(sparse(σ₊.data)) for σ₊ in σ₊_ops],
    c_σ₊_ops = [ComplexF64.(sparse(c_σ₊.data)) for c_σ₊ in c_σ₊_ops]
    );  

##Plotting functions

function n_seg2t_list(A::AbstractVector, dur, tlist)
    N = length(A)
    @assert length(dur) == N "dur must have the same length as A"

    t_edges = vcat(0.0, cumsum(dur))         # length N+1
    idx = searchsortedlast.(Ref(t_edges), tlist)  # 1..N+1
    idx = clamp.(idx, 1, N)                  # handle t == t_edges[end]

    return A[idx]
end

#this is for matrices
function n_seg2t_list(A::AbstractMatrix, dur, tlist)
    N = size(A, 1)  # number of rows
    @assert length(dur) == N "dur must have the same number of rows as A"

    t_edges = vcat(0.0, cumsum(dur))
    idx = searchsortedlast.(Ref(t_edges), tlist)
    idx = clamp.(idx, 1, N)

    return A[idx, :]          # pick rows
end

#theme of the figures
set_theme!(Theme(
    fontsize = 16,
    Axis = (
        xlabelsize = 22,
        ylabelsize = 22,
        titlesize  = 23,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xticksize = 8,
        yticksize = 8,
        tickwidth = 1.5,
    ),
    Legend = (
        labelsize = 14,
    )
))


function make_plot!(gp, v, tlist, title, yaxis; ylims = nothing)

    ax = Axis(
        gp,
        xlabel = L"\text{Dimensionless time} \; \;  [t g]",
        ylabel = yaxis,
        title  = title
)
if !isnothing(ylims)
    CairoMakie.ylims!(ax, ylims)
end
    if ndims(v) > 1

        linestyles = [:dash, :dot]
        colors = [:red, :blue]
        for rail in 1:N_rails
            lines!(
                ax,
                tlist,
                v[rail, :],
                linewidth = 3,
                label = L"i = %$rail",
                linestyle = linestyles[rail],
                color = colors[rail]
            )
        end
        #Legend(fig[1, 2], ax)
    return ax
    end

    if ndims(v) == 1
    lines!(
        ax,
        tlist,
        v,
        linewidth=3,
        color = :green
    )
    return ax
    end
    #axislegend(ax; position = :rc)
    
end

#Plots that where relevant during my thesis
function make_main_plots(exp_vals, ΩMatrix, ϵMatrix, tlist_solution)

    f1 = Figure(size = (900, 800))

    ax1 = make_plot!(f1[1, 1], exp_vals.exp_n_rail_cavity, tlist_solution, L"\langle c_i^\dagger c_i(t) \rangle \;\; \text{for cavity} \; \text{(optimized parameters)}", L"\langle n \rangle \; (\mathrm{population})"; ylims = [0, 1])
    ax2 = make_plot!(f1[1, 2], exp_vals.exp_rail_g1, tlist_solution, L"\langle |g_1 \rangle \langle  g_1 | (t)   \rangle \;\; \; \text{(optimized parameters)}", L"\; (\mathrm{population})"; ylims = [0, 1.1])
    hlines!(ax2, [0.5], color = :black, linestyle = :dash)
    ax3 = make_plot!(f1[2, 1], exp_vals.exp_rail_g2, tlist_solution, L"\langle |g_2 \rangle \langle  g_2 | (t)   \rangle \;\; \; \text{(optimized parameters)}", L"\; (\mathrm{population})"; ylims = [0, 1])
    hlines!(ax3, [0.5], color = :black, linestyle = :dash)
    ax4 = make_plot!(f1[2, 2], exp_vals.exp_rail_e, tlist_solution,  L"\langle |e \rangle \langle  e | (t)   \rangle \;\; \; \text{(optimized parameters)}", L"\; (\mathrm{population})"; ylims = [0, 1])
#Legend(f[1:2, 3], ax1, "Global Legend", framevisible = false)

    axislegend(ax1, position = :rt, framevisible = false)

    f2 = Figure(size = (900, 800))

    ax5 = make_plot!(f2[1, 1], real.(ϵMatrix)', tlist_solution, L"\text{Re} (\mathcal{E}) \   \ \text{(optimized parameters)}", L"\text{Scaled } \; \; ")
    ax6 = make_plot!(f2[1, 2], imag.(ϵMatrix)', tlist_solution, L"\text{Im}(\mathcal{E}) \   \ \text{(optimized parameters)}", L"\text{Scaled } \; \; ")
    ax7 = make_plot!(f2[2, 1], real.(ΩMatrix)', tlist_solution, L"\text{Re}(\Omega) \   \ \text{(optimized parameters)}", L"\text{Scaled } \; \; ")
    ax8 = make_plot!(f2[2, 2], imag.(ΩMatrix)', tlist_solution, L"\text{Im}(\Omega) \   \ \text{(optimized parameters)}", L"\text{Scaled } \; \; ")

    axislegend(ax5, position = :rt, framevisible = false)

    f3 = Figure(size = (900, 800))

    ax9 = make_plot!(f3[1, 1], exp_vals.exp_c_n_Total, tlist_solution, L"\langle \Sigma_i c_i^\dagger c_i | (t)   \rangle \;\; \;", L"\langle n \rangle \; (\mathrm{population})"; ylims = [0, 1])
    ax10 = make_plot!(f3[1, 2], exp_vals.exp_exc_n_Total, tlist_solution, L"\langle \Sigma_i c_i^\dagger c_i(t) + \langle |e_i \rangle \langle  e_i |  | (t)   \rangle \;\; \; ", L"\; (\mathrm{population})"; ylims = [0, 2])
    #hlines!(ax2, [0.5], color = :black, linestyle = :dash)
    ax11 = make_plot!(f3[2, 1], exp_vals.fid_time, tlist_solution, L"F(t) \;\; \;", L"\; (\mathrm{fidelity})"; ylims = [0, 1.1])
    hlines!(ax11, [1.0], color = :black, linestyle = :dash)
    ax12 = make_plot!(f3[2, 2], exp_vals.Tr_ρ_square, tlist_solution, L"Tr(\rho^2) (t)",  L"\; (\text{purity})"; ylims = [0, 1.1])
    hlines!(ax12, [1.0], color = :black, linestyle = :dash)
    #Legend(f[1:2, 3], ax1, "Global Legend", framevisible = false)


    return (f1 = f1,  f2 = f2, f3 = f3)
end
