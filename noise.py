#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from construct_H import basis_sort, create_binaries
from simulation import create_samples, analytic_ss
import importlib
# importlib.reload(simulation)


# In[3]:


def get_combos(n):
    
    combos = basis_sort(create_binaries(n))
    
    return combos


# In[106]:


def down_sampled(k_ons, k_off, epsilon, noise, nCells):
    '''
    Simulate cells using Ising model, where sites have on-rates k_ons, and return the states after binomial dropout.
    
    Args: 
        k_ons (list of floats): on rate at each site
        k_off (float): off rate at all sites
        epsilon (float): inverse correlation parameter
        noise (float): fraction of ATAC reads lost
        nCells (int): number of cells to sample
        
    Returns:
        samples (array): array with new gene-states per cell
    '''
    
    n = len(kons)
    N = 2**n
    
    # ATAC only simulation.
    bs = [0]*n
    ds = [0]*n
    
    simulation_time =  5/min(k_ons)
    
    # Take gene-state information only, at final time point.
    samples = create_samples(k_ons, k_off, bs, ds, epsilon, nCells, simulation_time)[1][:,-1,:-n]
    combos = get_combos(n)
    state_indices = [list(cell).index(1) for cell in samples]
    
    site_states = [combos[i] for i in state_indices]
    
    new_states = []
    # Apply binomial drop-out at each site.
    for state in site_states:
        new_state = state.copy()
        for j in range(len(state)):
            kept = np.random.binomial(1, 1-noise)
            if not kept:
                new_state[j] = 0
        new_states += [new_state]
    
    return site_states, new_states    


# In[92]:


def dist_from_samples(samples):
    '''
    Return probability distribution of form {(combo): prob} given list of samples.
    e.g. {(0,0): 0.5, (0,1): 0.2, (1,0): 0.2, (1,1):0.1}
    '''
    n = len(samples[0])
    combos = [tuple(i) for i in get_combos(n)]
    
    data_tuples = [tuple(item) for item in samples]
    counter = Counter(data_tuples)
    
    total_cells = sum(counter.values())
    sample_dist = {}
    for combo in combos:
        sample_dist[combo] = counter[combo]/total_cells

#     sample_dist = {k:v/total_cells for k,v in counter.items()}
    
    return sample_dist


# In[13]:


# Define kons, koffs, epsilon, noise, nCells
# ############
# imports
# 	numpy, construct H, simulator
# 		simulator might be simpler -- just size-N CTMC
# def fun():
# 	that computes prob change

# ############
# Example usage:
# simulate to give steady-state distribution samples (by Gillespie) -> N x nCells
# 	Binomial downsample (matrix as n argument for binomial rv)
# propagate H to get the steady-state P 2^N
# 	manipulate P using noise matrix/ etc to get P~ (downsampled)

# 1. sanity check: 
# 	take steady-state distribution, take its mean over cells, plot vs. noise * kons/(kons+koffs)
# 2. Cast simulation results to 2^N and plot vs. analytical distribution
# 	identity lines and maybe log-log
# 		maybe residuals as a marginal subplot
        
# others %%%%%%%%%
# computing the Ising solution
# computing the moments 

# Keep things specific to topics in the notebooks
# try to keep more generic things (simulator, H construction, casting between N and 2^N) in external py scripts


# In[76]:


def Ising_analytic_ss(k_ons, k_off, epsilon):
    
    n = len(k_ons)

    steady_state = analytic_ss(k_ons, k_off, epsilon)
    
    combos = get_combos(n)
    
    Ising_prob = {}
    
    for i in range(len(combos)):
        
        Ising_prob[tuple(combos[i])] = steady_state[i]    
        
    return Ising_prob

def Ising_analytic_drop(k_ons, k_off, epsilon, pdrop):
    
    new_dist = dist_drop(Ising_analytic_ss, [k_ons, k_off, epsilon], {}, pdrop)
    
    return new_dist


# In[65]:


def prob_change(string, drop):
    ''' 
    Calculates the probability flow from a given configuration to new configurations under drop-out.
    
    Args:
        string (tuple): The original configuration in the +1/-1 convention.
        drop (float): The drop-out probabilty for an individual site to flip from +1 to -1.
        
    Returns:
        dictionary: A dictionary whose keys are the possible resulting configurations after drop-out,
            and whose values are the probability of arriving at the new configuration from the old (and
            the negative of the probability of leaving the original configuration for the original configuration)
    '''
    
    # Identify how many 'on' sites in this configuration.
    on_indices = [i for i in range(len(string)) if string[i]==1]
    
    # Generate all possible drop-out scenarios. (e.g. all 'on' sites --> 'off', the second 'on' site only --> 'off', etc.)
    n = len(on_indices)
    flipped = []
    for i in range(2**n, 2**(n+1)):
        bin_string = bin(i)[3:]
        flipped += [[int(i) for i in bin_string]]
    
    # Calculate the probability of each drop-out scenario, and the new configuration which results.
    flipped_patterns = {}
    for f in flipped:
        prob = drop**sum(f)*(1-drop)**(n-sum(f))
        new_string = string.copy()
        for i in range(n):
            if f[i]:
                new_string[on_indices[i]] = 0
                
        # For the no drop-out scenario, report the negative of the probability of leaving the original configuration.
        if new_string==string:
            flipped_patterns[tuple(new_string)] = -(1 - prob)
        else:
            flipped_patterns[tuple(new_string)] = prob

    return flipped_patterns


def dist_drop(distribution, args, kwargs, drop):
    '''
    Modifies an analytic probabilty distribution with the effect of drop-out.  
    
    Args:
        distribution (function): a function which takes args and kwargs and returns a probability distribution
            in the form of a dictionary, with configuration tuples as keys and probabilites as values.
        args (list): the arguments for the analytic distribution.
        kwargs (dictionary): the key-word arguments for the analytic distribution.
        drop: the drop-out probability.
    '''
    
    # Start with an analytic distribution.
    current_prob = distribution(*args, **kwargs)
    new_prob = current_prob.copy()
    
    # For each configuration, calculate the probability flow.
    for tpl in current_prob.keys():
        string = list(tpl)
        flow = prob_change(string, drop)
        
        # Add the flow from the current string to the new string.
        for new_string, change in flow.items():
            new_prob[tuple(new_string)] += change*current_prob[tpl]
            
            if new_prob[tuple(new_string)] < 0:
                print('neg prob:', new_prob, )
        
    return new_prob


# In[ ]:


if __name__=='__main__':
    
    ## Show original simulated distribution and downsampled version.
    
    k_ons, k_off, epsilon, noise, nCells = [1,2], .2, 0.5, 0.5, 100000
    a = down_sampled(k_ons, k_off, epsilon, noise, nCells)
    
    # Plot samples and then downsampled distribution.
    data = a[0]
    data2 = a[1]

    # Convert lists to tuples for hashability
    data_tuples = [tuple(item) for item in data]
    data2_tuples = [tuple(item) for item in data2]

    # Count the frequency of unique tuples for each dataset
    counter = Counter(data_tuples)
    counter2 = Counter(data2_tuples)

    # Extract unique tuples and their counts for each dataset
    unique_elements = list(counter.keys())
    unique_elements2 = list(counter2.keys())

    # Combine unique elements from both datasets
    all_unique_elements = list(set(unique_elements + unique_elements2))

    # Get counts for each dataset
    counts = [counter[element] for element in all_unique_elements]
    counts2 = [counter2[element] for element in all_unique_elements]

    # Plotting the side-by-side bar chart
    bar_width = 0.35  # Adjust the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bar1 = ax.bar(
        range(len(all_unique_elements)),
        counts,
        width=bar_width,
        label='Simulated'
    )

    bar2 = ax.bar(
        [i + bar_width for i in range(len(all_unique_elements))],
        counts2,
        width=bar_width,
        label='Simulated Downsampled'
    )

    ax.set_xlabel('Site Configurations')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of Configurations')
    ax.set_xticks([i + bar_width / 2 for i in range(len(all_unique_elements))])
    ax.set_xticklabels(all_unique_elements, rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()


# In[ ]:


def get_means(distribution):
    combos = [i for i in distribution.keys()]
    n = len(list(combos[0]))

    means = []
    for i in range(n):
        mean = sum([combo[i]*key for combo, key in distribution.items()])
        means += [mean]
        
    return means

if __name__=='__main__':
    ## Check that dropped analytic solution has the expected means.

    analytic_dropped = Ising_analytic_drop(k_ons, k_off, epsilon, noise)

    means = get_means(analytic_dropped)

    analytic_original = Ising_analytic_ss(k_ons, k_off, epsilon)

    expected_means = [(1-noise)*mean for mean in get_means(analytic_original)]

    plt.title('Site Means: Calculated vs Expected')
    plt.scatter(expected_means, means)
    plt.plot(expected_means, expected_means, linestyle=':')
    plt.show()

    ## Check that entire distribution matches.

    simulated_dropped = dist_from_samples(a[1])
    keys = [i for i in simulated_dropped.keys()]
    x = [simulated_dropped[i] for i in keys]
    y = [analytic_dropped[i] for i in keys]

    plt.scatter(x,y)
    plt.plot(x,x)
    # plt.xscale('log')
    # plt.yscale('log')

    plt.title('Calculated vs Simulated Probabilities for each gene-state')
    plt.show()

    ## Plot distribution residuals.
    residuals = [(i-j)/(i+j) for i, j in zip(x,y)]
    plt.scatter([i for i in range(len(x))], residuals)
    plt.title('Residuals')
    plt.show()

