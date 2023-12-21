#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/pachterlab/GYP_2022/blob/dev/gg220909_telegraph.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Simulating switching processes coupled to non-Markovian processing
# 
# This notebook uses a modified version of the Gillespie algorithm to simulate a telegraph model of gene switching coupled to a mixed Markovian/non-Markovian model of mRNA splicing and degradation. We implement these simulations to validate generating function-based solutions to these systems.

# # Helper functions

# In[2]:


import numpy as np
from numpy import matlib
import scipy.stats

import import_ipynb
from construct_H import construct_H, basis_sort, create_binaries

import numba
import tqdm
import multiprocessing
import scipy
from scipy.fft import ifft, ifft2, ifftn, irfft, irfftn


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import time
t1 = time.time()


# ## Simulation

# In[3]:


@numba.njit
def update_rule(population, *args):
    '''
    Update rule that takes params and 
    updates the population matrix
    '''
    k_on, k_off, k, b1 = args
    m0, m1, m2 = population    
    flux =np.zeros(4)
    flux[0], flux[1], flux[2], flux[3] = k_on*(1-m0), k_off*m0, k*m0, b1*m1
    return flux
    
def sample_rxns(prob_arr):
    '''
    Takes prob_arr and
    returns an randomly sampled index according to 
    the probabilities specificed in prob_arr
    '''
    num = np.random.rand()
    sent = 0
    tot = 0.
    while tot < num:
        tot = tot + prob_arr[sent]
        sent = sent + 1
    return sent - 1

def step(calc_flux, population, t, *args):
    '''
    Takes flux calculating function, past event, 
    flux, population, t, args and
    returns time to reaction and the reaction that happens
    '''
    flux = calc_flux( population, *args)[0]
    flux_sum = np.sum(flux)
    t = np.random.exponential(1. / flux_sum)
    rxn_prob = flux/flux_sum
    rxn = sample_rxns(rxn_prob)
    
    return rxn, t
def step_with_queued(calc_flux,population,t,queued_reactions,queued_reaction_times,export_arr,*args):
    '''
    Takes flux calculating function, past event, 
    flux, population, t, args and
    returns time to reaction and the reaction that happens
    '''
    # return step(calc_flux,population,t,*args)
    if len(queued_reactions)==0:
        return step(calc_flux,population,t,*args)
    else:
        # 
        population = np.copy(population)
        population_orig = np.copy(population)
        # queued_reactions = np.copy(queued_reactions)
        queued_reaction_times = np.copy(queued_reaction_times)

        init_flux = calc_flux( population, *args)[0]
        n_rxn = len(init_flux)
        fluxes = np.zeros((len(queued_reactions)+1,n_rxn))
        fluxes[0] = calc_flux( population, *args)[0]
        for q in range(len(queued_reactions)):
            population += export_arr[queued_reactions[q]]
            fluxes[q+1] = calc_flux( population, *args)[0]

        queued_reaction_times = np.concatenate(([t],queued_reaction_times))
        flux_sums = fluxes[:-1] * np.diff(queued_reaction_times)[:,None]
        flux_sums = np.concatenate((np.zeros(n_rxn)[None,:],flux_sums))
        flux_cumsums = np.cumsum(flux_sums,0)
        tot_flux_cumsum = flux_cumsums.sum(1)

        u = np.random.rand()
        flux = np.log(1/u)
        
        last_ind = np.where(tot_flux_cumsum<flux)[0][-1]
        dt = queued_reaction_times[last_ind]-t
        flux_orig = np.copy(flux)
        flux -= tot_flux_cumsum[last_ind]
        dtprime = flux/fluxes[last_ind].sum()
        dt += dtprime

        flux_sum = fluxes[last_ind]
        rxn_prob = flux_sum / flux_sum.sum()
        rxn = sample_rxns(rxn_prob)   
        return rxn, dt

def markovian_simulate(calc_flux, update, population_0, t_arr, tau, \
                       DELAYED_SPECIES_GENERATORS, export_arr,tau_list=None,\
                       return_full=False,*args):
    '''
    Takes params and
    returns matrix of the number of counts for 
    a species j at time t[i]
    '''
    if isinstance(tau , list) or isinstance(tau , tuple) :
        n_delays = len(tau)
    else:
        tau = [tau]
        n_delays = 1

    population_arr = np.zeros((len(t_arr), update.shape[1]))
    t = t_arr[0]
    population = population_0.copy()
    population_arr[0, :] = population
    t_list = []
    population_list = []
    rxn_list = []

    t_list.append(t)
    population_list.append(population)

    # Delay time queue
    if tau_list is None:
        tau_list = [[]]*n_delays
    tau_list_delay_inds = np.asarray(sum([[j]*len(x) for j,x in enumerate(tau_list)],[]))
    tau_list_all_delays = np.concatenate(tau_list)
    tau_list_sort = np.argsort(tau_list_all_delays).astype(int)

    tau_list_delay_inds = list(tau_list_delay_inds[tau_list_sort])
    tau_list_all_delays = list(tau_list_all_delays[tau_list_sort])
    
    while t < t_arr[-1]:
        event,dt = step_with_queued(calc_flux,population,t,\
                                    tau_list_delay_inds,tau_list_all_delays,\
                                    export_arr, *args)
        

        temp_t = t + dt
        
        # Check if an export event should slot in before t + dt
        while (len(tau_list_all_delays) != 0) and (temp_t > tau_list_all_delays[0]):
            population = population + export_arr[tau_list_delay_inds[0]]
            t = tau_list_all_delays[0]
            if t>t_arr[-1]:
                break
            t_list.append(t)
            population_list.append(population)
            rxn_list.append(-tau_list_delay_inds[0]-1)
            tau_list_all_delays.pop(0) # remove element
            tau_list_delay_inds.pop(0)

        if len(tau_list_all_delays)!=len(tau_list_delay_inds):
            raise ValueError
        
        # Perform event
        population = population + update[event, :]
        t = temp_t
        t_list.append(t)
        population_list.append(population)
        rxn_list.append(event)


        for j in range(n_delays):
            if event in DELAYED_SPECIES_GENERATORS[j]:
                tau_list_all_delays.append(t + tau[j])
                tau_list_delay_inds.append(j)
        tau_list_sort = np.argsort(tau_list_all_delays).astype(int)
        tau_list_delay_inds = list(np.asarray(tau_list_delay_inds)[tau_list_sort])
        tau_list_all_delays = list(np.asarray(tau_list_all_delays)[tau_list_sort])
    
    t_only_arr = np.asarray(t_list)
    population_list= np.asarray(population_list)
    for i in range(len(t_only_arr)):
        ind = np.searchsorted(t_arr, t_only_arr[i], side = 'right')
        population_arr[ind:] = np.array(population_list[i])
    if return_full:
        return population_arr,t_only_arr,population_list,rxn_list
    else:
        return population_arr


# ## Simulation setup 
# 

# In[4]:


def nondelay_wrapper(N,n,K_val,A_val,beta_val,gamma_val,initcond,number_of_cells=5000,simulation_time=5,ndel=50):

    H,A,C,S,k,Hss = sim_setup(N,n,K_val,A_val,beta_val,gamma_val)

    delayed_species = []
    DELAYED_SPECIES_GENERATORS = [[]]

    tau = []

    ####
    update_arr = S

    t_arr = np.linspace(0, simulation_time, ndel,endpoint=False)

    export_arr = np.zeros((len(delayed_species),N+n),dtype=int)

    samples = np.zeros((number_of_cells, len(t_arr), N+n))

    Hss = scipy.linalg.null_space(H.T)
    Hss /= Hss.sum()
    Hss = Hss.squeeze()

    for i in tqdm.tqdm(range(number_of_cells)):
        population_0 = np.zeros(N+n,dtype=int)
        if initcond[0][0]=='thermalized':
            init_state_prob = Hss
        elif isinstance(initcond[0][1],int):
            init_state_prob = np.zeros(N)
            jinit = initcond[0][1]
            init_state_prob[jinit] = 1
        else:
            init_state_prob = initcond[0][1]
        if N>1:
            population_0[:N] = np.random.multinomial(1,init_state_prob,1)
        else:
            population_0[:N] = 1
        initial_promoter_state = np.where(population_0[:N]==1)[0][0]

        for j,condition in enumerate(initcond[1:]):
            if isinstance(condition[0],str):
                if condition[0]=='deterministic':
                    population_0[j+N] = condition[1]
                elif condition[0]=='poisson':
                    population_0[j+N] = np.random.poisson(condition[1])
            else:
                if condition[initial_promoter_state][0]=='deterministic':
                    population_0[j+N] = condition[initial_promoter_state][1]
                elif condition[initial_promoter_state][0]=='poisson':
                    population_0[j+N] = np.random.poisson(condition[initial_promoter_state][1])

        samples[i, :, :],event_t,event_x,_ = markovian_simulate(
            propfun_generic, update_arr, population_0, t_arr, \
            tau, DELAYED_SPECIES_GENERATORS, export_arr,[[]],\
            True,(k,K_val,A_val,beta_val,gamma_val,N))
    return t_arr,samples,H,A,C,init_state_prob


# In[6]:


def pss_wrapper_n(H_,A,C,t,mx,n,N,initcond):
    
    g = np.asarray(get_g(mx)).T
    L,V,Vinv = compute_eigs(C)
    coeff = np.asarray([compute_coeff(L,V,Vinv,g.T,i) for i in range(n)])
    
    if initcond[0][0]=='thermalized':
        Hfin = scipy.linalg.null_space(H_.T)
        Hfin /= Hfin.sum()
        w = Hfin.squeeze()
    elif initcond[0][0]=='deterministic':
        w = np.zeros(N)
        w[initcond[0][1]] = 1
    elif initcond[0][0]=='categorical':
        w = initcond[0][1]
    else:
        raise ValueError('Not implemented')
    w = np.asarray(w,dtype=np.complex128)

    gf_initcond_prop = np.ones((g.shape[0],N),dtype=np.complex128)
    
    for j,condition in enumerate(initcond[1:]):
        if isinstance(condition[0],str): #Identical conditions for all states
            if condition[0]=='deterministic':
                initcond_gfun = lambda x: (x+1)**condition[1]
            elif condition[0]=='poisson':
                initcond_gfun = lambda x: np.exp(condition[1]*x)
            for i in range(g.shape[0]):
                gf_initcond_prop[i,:] *= initcond_gfun(ufun_generic(t,L,coeff[j,...,i]))
        elif len(condition)==N: #list of length N
            for k in range(N):
                if condition[k][0]=='deterministic':
                    initcond_gfun = lambda x: (x+1)**condition[k][1]
                elif condition[k][0]=='poisson':
                    initcond_gfun = lambda x: np.exp(condition[k][1]*x)
                for i in range(g.shape[0]):
                    gf_initcond_prop[i,k] *= initcond_gfun(ufun_generic(t,L,coeff[j,:,i]))
        else: 
            print(N)
            print(len(condition[0]))
            raise ValueError
            
    gf_initcond_prop *= w 
    gf = gfun_numerical(g,t,H_,A,N,L,coeff,gf_initcond_prop)
#     Pss = np.zeros((mx[0], mx[1],N))
    Pss = np.zeros(mx + [N])
    
    
    for j in range(N):
        Pss[..., j] = np.real(ifftn(np.reshape(gf[:,j], (mx))))
#         Pss[:,:,:,j] = np.real(ifftn(np.reshape(gf[:,j], (-1,mx[1],mx[2]))))
    return Pss.squeeze()

def gfun_numerical(g,t,H,A,N,L,coeff,gf_initcond):
    gf_ = np.zeros((g.shape[0],N),dtype=np.complex128)
    
    for i in (range(g.shape[0])):
        t0 = 0
        y0 = gf_initcond[i]
        
        Ufun = lambda x: np.asarray([ufun_generic(x,L,coef[:,i]) for coef in coeff])
        intfun = lambda t,y: intfun_multi(t,y,H,A,Ufun)
        res = scipy.integrate.solve_ivp(intfun,[t,t0],y0)
        while res.status == 'running':
            res.step()
        gf_[i] = res.y[:,-1]
    return gf_


# In[7]:


def sim_setup(N,n,K_val,A_val,beta_val,gamma_val):

    H = np.zeros((N,N))
    for kv in K_val:
        H[kv[0],kv[1]] = kv[2]
    H -= np.diag(H.sum(1))

    A = np.zeros((N,n))
    for av in A_val:
        A[av[0],av[1]-1] = av[2]

    B = np.zeros((n,n))
    for bv in beta_val:
        B[bv[0]-1,bv[1]-1] = bv[2]
    B -= np.diag(B.sum(1))
    for gv in gamma_val:
        B[gv[0]-1,gv[0]-1] -= gv[1]
    

    Nspec = N+n
    Nrxn = len(K_val) + len(A_val) + len(gamma_val) + len(beta_val)
    S_mark = np.zeros((Nrxn,Nspec))
    k_mark = np.zeros(Nrxn)

    i=0
    for kv in K_val:
        S_mark[i,kv[0]] = -1
        S_mark[i,kv[1]] = 1
        k_mark[i] = kv[2]
        i+=1

    for av in A_val:
        S_mark[i,av[1]+N-1] = 1
        k_mark[i] = av[2]
        i+=1
        
    for bv in beta_val:
        S_mark[i,bv[0]+N-1] = -1
        S_mark[i,bv[1]+N-1] = 1
        k_mark[i] = bv[2]
        i+=1
        
    for gv in gamma_val:
        S_mark[i,gv[0]+N-1] = -1    
        k_mark[i] = gv[1]
        i+=1

    Hss = scipy.linalg.null_space(H.T)
    Hss /= Hss.sum()
    Hss = Hss.squeeze()

    S_mark = S_mark.astype(int)

    return H,A,B,S_mark,k_mark,Hss

def propfun_generic(x,*args):
    k,K_val,A_val,beta_val,gamma_val,N = args[0]
    nRxn = len(k)
    nCells = x.shape[0]

    a = np.zeros((nCells,nRxn),dtype=float)
    x = x.T
    a = a.T
    #######
    j = 0 
    for i in range(len(K_val)):
        a[j] = k[j] * x[K_val[i][0]]
        j+=1
    for i in range(len(A_val)):
        a[j] = k[j] * x[A_val[i][0]]
        j+=1
    for i in range(len(beta_val)):
        a[j] = k[j] * x[beta_val[i][0]+N-1]
        j+=1
    for i in range(len(gamma_val)):
        a[j] = k[j] * x[gamma_val[i][0]+N-1]
        j+=1
    # raise ValueError
    return a.T


# ## PGF computation for generic non-delayed systems

# In[8]:


def get_g(mx):
    u = []
    for i in range(len(mx)):
        l = np.arange(mx[i])
        u_ = np.exp(-2j*np.pi*l/mx[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()
    return g

def compute_coeff(L,V,Vinv,u,j=0):
    n_u = u.shape[1]
    a = np.asarray([(V@np.diag( Vinv @ u[:,i]))[j] for i in range(n_u)]).T
    return a

def compute_eigs(C):
    L,V = np.linalg.eig(C)
    Vinv = np.linalg.inv(V)
    return L,V,Vinv


# In[9]:


def ufun_generic(x,L,coeff):
    Ufun = (np.exp(L*x)@coeff)
    return Ufun

def intfun_multi(t,y,H,A,Ufun):
    dy = np.dot(H.T,y) + np.dot(A,Ufun(t))*y
    return -dy


# ## Visualization for generic non-delayed systems

# In[19]:


# conditional_colors = ('red','cadetblue','deeppink','khaki')
conditional_colors = ('red','cadetblue','rebeccapurple','mediumseagreen', 'red','cadetblue','rebeccapurple','mediumseagreen')
uncond_colors = ('lightgray','darkgray')
alf=0.5

def remove_neg(Pss):
    Pss[Pss<0] = min(Pss[Pss>0])
    Pss = Pss/np.sum(Pss)
    return Pss


# In[26]:


def viz_nondelay2(t_arr,samples,N,n,H,A,C,initcond,init_state_prob):
    
    fig1,ax1=plt.subplots(n+1,max(N, n+1),figsize=(12,12))
    
    # Get means over time.
    gene_means = {}
    for gene in range(n):
        
        cond_means = []
        uncond_means = []
        
        for t in range(len(t_arr)):
            snapshot = samples[:, t, :]

            cond_means_list = []
            
            for j in range(N):
                cf = snapshot[:,j]==1
                cond_mean = snapshot[cf,N+gene:N+gene+1].mean(0)
                cond_means_list += [cond_mean]
                
            cond_means += [cond_means_list]

            uncond_mean = snapshot[:,N+gene:N+gene+1].mean(0)
            uncond_means += [uncond_mean]
            
        gene_means[gene] = [cond_means, uncond_means]
    
    for i in range(n):
        for j in range(N):
            ax1[0,i].plot(t_arr, [p[j] for p in gene_means[i][0]], color=conditional_colors[j])
            
        ax1[0,i].set_title('Mean Species '+str(i+1))
        ax1[0,i].set_xlabel('Time')

#     ax1[0,0].plot(t_arr, [i[1] for i in mean1s])
#     ax1[0,1].plot(t_arr, [i[1] for i in mean2s])
# #     ax1[0,2].plot(t_arr, )
    
#     ax1[0,0].set_title('Mean Species 1')
#     ax1[0,1].set_title('Mean Species 2')
#     ax1[0,0].set_xlabel('Time')
#     ax1[0,1].set_xlabel('Time')
    
    
    # Plot final gene state distribution.
    tind = len(t_arr)-1
    final = samples[:,tind,:]
    X = final
    
    time = t_arr[tind]
    f = max(N, n+1)-1
    
    ax1[0,f].bar(np.arange(N),X[:,:N].mean(0),color='lightgray')
    Pss_gene = np.dot(scipy.linalg.expm(time*H.T),init_state_prob)
    ax1[0,f].plot(Pss_gene,'r-')
    ax1[0,f].set_xticks([])
    ax1[0,f].set_yticks([])
    ax1[0,f].set_ylabel('t = {:.2f}'.format(t))
    
    ax1[0,f].set_title('Final Gene State Probability')
    ax1[0,f].set_xlabel('State')
    
    # Get unconditional distribution.
#     if n==2:
    mx_uncond = [int(final[:,N+i].max() + 3) for i in range(n)]

    Pss_uncond = pss_wrapper_n(H,A,C,time,mx_uncond,n,N,initcond)
    margin = np.sum(Pss_uncond, axis=n)
        
    # Plot conditional distributions for each species:
    for i in range(n):  
        mx = [1]*n
        ub = final[:,N+i].max() + 6
        mx[i] = int(ub)
        Pss = pss_wrapper_n(H,A,C,time,mx,n,N,initcond)
        
        for j in range(N):
            cf = final[:,j]==1

            bins = np.arange(ub)-0.5

            hist,_ = np.histogram(final[cf,N+i],bins,density=True)

            ax1[1+i,j].bar(bins[:-1]+0.5,hist*Pss_gene[j],color=conditional_colors[j],alpha=alf)
            ax1[1+i,j].plot(Pss[:,j],color=conditional_colors[j],alpha=alf)
            
            
#             # Plot marginals from Pss:
#             if n==2:
#                 if i==0:
#                     axis =1
#                 elif i==1:
#                     axis=0
                    
            sum_axes = []
            for k in range(n):
                if k!=i:
                    sum_axes += [k]
            marginal1 = Pss_uncond[...,j].sum(axis=tuple(sum_axes))
            
            ax1[1+i,j].plot(marginal1, color='black')
            
            ax1[1,j].set_title('State '+str(j+1))
            
            ax1[1+i, 0].set_ylabel('Marginal: Gene ' + str(i+1))
            
#     fig1.suptitle('Epsilon = ' + str(epsilon), fontsize=20)
    fig1.tight_layout()
#     plt.savefig('epsilon_'+str(epsilon)+"_Ising.jpg", dpi=600)
    plt.show()  
            
    def rand_jitter(arr):
        stdev = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def jitter(x, y, s=None, c='b', marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
        return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)
    
 
    if n==2:
        
        plt.imshow(remove_neg(margin), cmap='hot', norm=LogNorm())
        plt.colorbar()
#         plt.title('Marginal Joint Distribution, Epsilon = ' + str(epsilon))
        plt.xlabel('Species 1')
        plt.ylabel('Species 2')
        
        jitter(final[:,N], final[:,N+1], s=1)
    
    return Pss_uncond


# # Scenarios

# Here, we consider systems in order of increasing complexity, and compare their solutions to simulations. "Homogeneous" means the system starts at zero molecules. "Thermalized" means its initial distribution over gene states is the steady state.

# ## 4-state, 2-species, homogeneous thermalized

# In[21]:


# Take k_ons, k_off, bs, ds, epsilon, n_samples
# Return steady-state simulation.


# In[22]:


def create_A_val(bs):
    '''
    Create the transcription information input, A_val, for nondelay_wrapper.
    
    Args:
        bs (list of floats): Transcription rate, b^i for each of n sites.
 
    Returns:
        list of lists: A_val input for nondelay_wrapper.
        A list of lists of the form: [state, species, transcription rate] (species are 1-indexed)
        e.g. for a two-gene system: [[1,1,b1], [2,2, b2], [3, 1, b1], [3, 2, b2]]
    '''
    
    n = len(bs)
    # Get possible on-off DNA configurations.
    combos = basis_sort(create_binaries(n))
    N = len(combos)
    # Create A, list of [state, species, rate]
    A_val = []
    for i in range(N):
        combo = combos[i]
        for j in range(n):
            # Species are 1-indexed.
            if combo[j]==1:
                A_val += [[i, j+1, bs[j]]]
    return A_val

def create_gamma_val(gammas):
    '''
    Create the decay information input for nondelay_wrapper, gamma_val.
    
    Args:
        gammas (list of floats): Decay rate, gamma^i for each of n species.
 
    Returns:
        e.g. [[2,d2], [1,d1]]
    '''
    n = len(gammas)
    gamma_val = [ [i+1, gammas[i]] for i in range(n)]
    return gamma_val

def K_val_from_H(H):
    K_vals = []
    for i in range(len(H)):
        for j in range(len(H[0])):
            if i!=j:
                K_vals += [[i, j, H[i][j]]]
    return K_vals


# In[23]:


def create_samples(k_ons, k_off, bs, ds, epsilon, number_of_cells, simulation_time):
    '''
    Use Markovian simulation to return simulated cells according to an Ising model.
    '''
    n = len(bs)
    N = 2**n
    
    H = construct_H(k_ons, k_off, epsilon, periodic=False)
    K_val = K_val_from_H(H)
    beta_val = []
    A_val, gamma_val = create_A_val(bs), create_gamma_val(ds)
    
    initcond = [['thermalized']] + [['deterministic',0]]*n
    
    t_arr,samples,H,A,C,init_state_prob = nondelay_wrapper(N,n,K_val,A_val,beta_val,gamma_val,initcond, simulation_time=simulation_time, number_of_cells=number_of_cells)
    return t_arr,samples,H,A,C,init_state_prob, initcond

def analytic_ss(k_ons, k_off, epsilon):
    
    H = construct_H(k_ons, k_off, epsilon, periodic=False)

    Hss = scipy.linalg.null_space(H.T)
    Hss /= Hss.sum()
    Hss = Hss.squeeze()
    
    return Hss

def visualize(k_ons, k_off, bs, ds, epsilon, number_of_cells, simulation_time):
    '''
    Visualize combined analytic and simulated results.
    Only suitable for max 3-gene systems. Recommended for 2-gene system.
    '''
    n = len(bs)
    N = 2**n
    
    t_arr,samples,H,A,C,init_state_prob, initcond = create_samples(k_ons, k_off, bs, ds, epsilon, number_of_cells, simulation_time)
    
    a = viz_nondelay2(t_arr,samples,N,n,H,A,C,initcond,init_state_prob)          


# In[27]:

if __name__ == "__main__":
    k_ons, k_off, bs, ds, epsilon, number_of_cells = [1]*2, 2, [4]*2, [1]*2, 0.5, 100
    simulation_time = 5/min(k_ons)
    # n, N = len(bs), 2**(len(bs))
    a = create_samples(k_ons, k_off, bs, ds, epsilon, number_of_cells, simulation_time)

    visualize(k_ons, k_off, bs, ds, epsilon, number_of_cells)


# In[ ]:





# In[ ]:




