#ARGS[1] if it is 0 then run the optimization without loss, if it is 1 ADD loss.
#ARGS[2] this value corresponds to the κ value
#ARGS[3] this value corresponds to the γ value. This argument just makes sense if there is loss, otherwise if it is included without loss it will print an error message
#ARGS[4] the name of the file I want to start the initial guess


if length(ARGS) < 2 || length(ARGS) > 4
    println("Error: Wrong number of arguments.")
    exit(1) #stop the script
end

job_index = parse(Int, get(ENV, "LSB_JOBINDEX", "1"))




#load packages 
using QuantumToolbox, Zygote, ExponentialAction, SparseArrays, Optimization, OptimizationOptimisers, OptimizationOptimJL, JLD2, LinearAlgebra, FiniteDiff, ExponentialUtilities, Random, DSP

#define essential stuff
const N_fock =  3; #dimension for the Fock states
const N_rails = 2; #number of rails
const dt = .01; #dt
const n_seg = 20; #number of segments for the control fields
#const τmin = 2*dt; #minimum duration of a single segment
#const τmax = 100*dt; #total max time of τmax*n_seg = 40  (for now)
const t_wait = 2.0; #time where there is no control fields
const tF = 5.0;
const tlist = 0:dt:tF #IF I CHANGE THIS I NEED TO CHANGE THE KNOTS
const tlist_wait = 0:dt:t_wait
Ω_amplitude = 15; #amplitudes for the control signals. This means it is NOT the max value. The max value is this divided by two. 
ϵ_amplitude = 60;
Random.seed!(job_index)
#--------------------------#
#define operators I will use for the whole simulation process
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

#tensor the Fock representation with the Lambda system representation for a single state.
# Basically a unit of state will be |state⟩ = |Fock⟩ ⊗ |Λ⟩ 
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

#I need this list format for the construction of the Hamiltonian (is more convenient)
c_ops = [c_1, c_2]; 
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


#Collapse operators γ for Lindblad master equation
if ARGS[1] != "1"
    γ = [1.0, 1.0]; #for both cavities, this term defines the rate 
else
    γ_value = parse(Float64, ARGS[3])
    γ = [γ_value, γ_value]
  
end

collapse_ops = [sqrt(γ[1])*c_ops[1], sqrt(γ[2])*c_ops[2]]; #vector of the collapse operators, it is the anhilitation operators for each of the two cavities.

## Helper functions 

#Optimization task functions
function make_opt_problem(cost, u0, params)
    f_obj(u, p) = cost(u, p)

    optf = OptimizationFunction(f_obj, Optimization.AutoFiniteDiff())
  
    return OptimizationProblem(optf, u0, params) #lb = lower_bounds, ub = upper_bounds )
end

#calls the solver
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
        #sol_adam = Optimization.solve(prob, OptimizationOptimisers.Adam(adam_lr); maxiters = adam_iters, callback = callback)
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


## Definition of physics

#define constants from the Hamiltonian
const Δ_2 = 0; #detunings 
const Δ_1 = 0;
const Δ_c = 0;
const A = 6; #amplitudes for the control signals. This means it is NOT the max value. The max value is this divided by two. 
const B = 6;
const g = 1; #coupling between matter and light
const α = 0.3;

κ_value = parse(Float64, ARGS[2])

κ = [κ_value, κ_value]


C = [0 1; 1 0] #Matrix that appears from the SLH

#Matrices are actually constants so I can make this more efficient
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
## Evolutions
#H = create_H(H0, H_Ω, H_ϵ)


function create_H(H0, H_Ω, H_ϵ, Ω, ϵ) #now the matrices are constants and I do not need to recompute them every time I call the function

    H = H0 + sum(H_Ω[i][1]*real(Ω[i]) + H_Ω[i][2]*imag(Ω[i]) + H_ϵ[i][1]*real(ϵ[i]) + H_ϵ[i][2]*imag(ϵ[i]) for i in 1:N_rails) 
end

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

## 
H_terms = create_H_terms(κ, C)

H0 = H_terms.H0
H_Ω = H_terms.H_Ω
H_ϵ = H_terms.H_ϵ

L_terms = create_terms_L([collapse_ops[1].data, collapse_ops[2].data]);

L_H0 = L_terms.L_H0
L_HΩ = L_terms.L_HΩ
L_Hϵ = L_terms.L_Hϵ
L_diss = L_terms.L_diss

ψ0 = tensor(basis(N_fock, 0), g_1, basis(N_fock, 0), g_1); #[|0) ⊗ |g_1)] ⊗ [|0) ⊗ |g_1)]
ψ0_QO = ψ0
ψ0 = ψ0.data

#desired state (notice that it is just defined in the matter system)
ψT_matter = (tensor(g_1, g_1) + tensor(g_2, g_2))/sqrt(2);
ψT_matter = ψT_matter.data;

pulse = tukey(n_seg, α) 



function Hamiltonian_evolution(ψ0::AbstractVector{<:Complex}, H0, H_Ω::AbstractVector, H_ϵ::AbstractVector, Ω, ϵ, τ,t_wait)
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
    return ρ.data
end


#partial traces (for pure states and mixed states)
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
##Creation of the control fields given the parameters 
#for a given input parameters it outputs the actual (Ω values for the system)
#because remember than the parameters are different from the actual values that we are inputting to the system
#the outputs of the functions is what we put as an input to the Hamiltonian 

function create_ΩMatrix(p::AbstractVector{<:Real}, A::Real, pulse)
    #p is a vector containing the matrix components of all the parameters for the Ω terms (all the rails and all the segments)
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


## Cost functions 
#this considers Schrodinger evolution 
function cost_base(u, params)
    
    (; Ω_amplitude, ϵ_amplitude, ψ0, t_wait, ψT_matter, κ, H0, H_Ω, H_ϵ, τvector, pulse) = params
  
    #define the indexes for the functions
    idx_Ω = 1:2*n_seg*N_rails;
    idx_ϵ = 2*n_seg*N_rails+1 : 4*n_seg*N_rails;
  
    #define the things that will actually go into the Hamiltonian (these depend on the parameter p)
    ΩMatrix = create_ΩMatrix(view(u, idx_Ω), Ω_amplitude, pulse);
    ϵMatrix = create_ϵMatrix(view(u, idx_ϵ), ϵ_amplitude, pulse);


    ψf = Hamiltonian_evolution(ψ0, H0, H_Ω, H_ϵ, ΩMatrix, ϵMatrix, τvector, t_wait)

    ρm = partial_trace_cavities(ψf)
    Fsum = real(ψT_matter'*ρm*ψT_matter)

    return 1 - Fsum
end
#This considers a Master equation approach
function cost_master(u, params)
    (; Ω_amplitude, ϵ_amplitude, ψ0, t_wait, ψT_matter, κ, H0, H_Ω, H_ϵ , τvector, pulse) = params

    #define the indexes for the functions
    idx_Ω = 1:2*n_seg*N_rails;
    idx_ϵ = 2*n_seg*N_rails+1 : 4*n_seg*N_rails;
    
    #define the things that will actually go into the Hamiltonian (these depend on the parameter p)
    
    ΩMatrix = create_ΩMatrix(view(u, idx_Ω), Ω_amplitude, pulse);
    ϵMatrix = create_ϵMatrix(view(u, idx_ϵ), ϵ_amplitude, pulse);
    #CMatrix = create_C_matrix([0, 1, 1, 0]);
   
    ρf = Liouvillian_evolution_mesolve(ψ0, H0, H_Ω, H_ϵ, ΩMatrix, ϵMatrix, collapse_ops, τvector, t_wait)
    ρm = partial_trace_cavities_density(ρf)
    Fsum = real(ψT_matter'*ρm*ψT_matter)
    return 1-Fsum
end

##End of definining funcitions and stuff 

#u0 = ones(4*n_seg*N_rails) + 0.6*(rand(4*n_seg*N_rails) .- 0.5)
u0 =  0.6*(rand(4*n_seg*N_rails) .- 0.5)
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
    collapse_ops = collapse_ops,
    L_H0 = L_H0,
    L_HΩ = L_HΩ,
    L_Hϵ = L_Hϵ,
    L_diss = L_diss,
    τvector = tF/n_seg*ones(n_seg),
    ψ0_QO = ψ0_QO,
    pulse = pulse,
);

## 

save_dir = "random_iterations" #define the folder
mkpath(save_dir) 

if length(ARGS) == 4
    if ARGS[4] == "yes"
        κ = κ[1]  
        path_file = joinpath("results", "sol_H_$(κ)_$(job_index)_v11.jld2")
    else
        name_input = ARGS[4]
        path_file = joinpath("results", name_input)
    end
    if isfile(path_file)
        data = load(path_file)["results_to_save"]
        u0 = data.u
        println("Optimization initialized with a personalized initial guess")
    else
        @error "File not found at $path_file."
        exit(1)
    end
end
    
if ARGS[1] == "0"
    println("Starting optimization without loss with κ=$(κ[1])")

    sol = solve_problem(cost_base, u0, params); #OPTIMIZATION

    cost_value = sol.objective
    κ = κ[1]           
    #save  file
    file_name = "sol_H_$(κ)_$(job_index)_v11.csv"
    full_path = joinpath(save_dir, file_name)

    open(full_path, "w") do file
    #
        println(file, "$job_index, $cost_value, $κ")
    end

    results_to_save = (
    u = sol.u,   
    objective = sol.objective, 
    stats = sol.stats,
    κ = κ           
    )

    file_name_data = "sol_H_$(κ)_$(job_index)_v11.jld2"

elseif ARGS[1] == "1"
    println("Starting optimization with loss with κ=$(κ[1]) and  γ=$(γ[1])")
    sol = solve_problem(cost_master, u0, params, adam_iters=0, lbfgs_iters=500); #OPTIMIZATION
  
    cost_value = sol.objective
    κ = κ[1]
    γ = γ[1]     
    

    file_name = "sol_L_kappa=$(κ)_gamma=$(γ)_$(job_index)_v11.csv"
    full_path = joinpath(save_dir, file_name)

    open(full_path, "w") do file
    #
        println(file, "$job_index, $cost_value, $κ, $γ")
    end

    results_to_save = (
    u = sol.u,   
    objective = sol.objective, 
    stats = sol.stats,
    γ = γ           
    )

    file_name_data = "sol_L_kappa=$(κ)_gamma=$(γ)_$(job_index)_v11.jld2"

else 
    print("Argument 1 is not valid")
    exit(1)
end

mkpath("results")
full_path_data = joinpath("results", file_name_data)
jldsave(full_path_data; results_to_save)
#save and confirm 
#println("finished and results saved with name: $file_name")
