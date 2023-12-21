import functools 
import numpy as np
import scipy
import argparse

def create_binaries(n):
    '''
    Create all possible binary strings of length n.
    '''
    combos = []
    for i in range(2**n, 2**(n+1)):
        bin_string = bin(i)[3:]
        combos += [[int(i) for i in bin_string]]
    return combos

def compare(string1, string2):
    ''' 
    Determine if string1 should precede string2 in desired basis.
    
    '''
    n = len(string1)
    
    if sum(string1) < sum(string2):
        return 1
    elif sum(string1) > sum(string2):
        return -1
    
    else:
        for i in range(n):
            if string1[i] > string2[i]:
                return 1
            elif string1[i] < string2[i]:
                return -1
    return 0

def basis_sort(combos):
    '''
    Returns list of strings ordered by sum and then by position of first 1 value.
    '''
    key=functools.cmp_to_key(compare)
    return sorted(combos, key=key, reverse=True)

def count_mismatch(string, periodic=False):
    '''
    Count number of misaligned pairs in a string.
    '''
    count = 0
    for i in range(len(string)-1):
        if string[i] != string[i+1]:
            count +=1
    if periodic:
        if string[-1] != string[0]:
            count += 1
    return count


def construct_H(k_ons, k_off, epsilon, periodic=False):
    '''
    Construct Markovian transition matrix H for an Ising model of chromatin.
    
    @params: 
        k_ons (list): List of rates of switching on for each of the n sites in the system.
        k_off (float): Rate of switching off (same for all sites)
        epsilons (float): The factor by which probability for strings with misaligned neighboring sites are 
        penalized. (epsilon<1 for positive correlation.)
        
    @returns:
        H (matrix): 2^n dimensional square matrix with Ising distribution for Null(H^T)
    '''
    
    n = len(k_ons)
    N = 2**n
    
    # Generate all binary strings of length N, ordered by sum and then by position of first 1 value.
    combos = basis_sort(create_binaries(n))
    
    H = np.zeros((N, N))
    
    # Iterate through combinations and their transitions to other strings.
    for i in range(N):
        current = combos[i]
        num_mismatch = count_mismatch(current, periodic=periodic)
        
        flows_out = []
        
        for j in range(n):
            new = current.copy()
            
            if current[j] == 0:
                new[j] = 1
                # Find index of new string in basis.
                new_index = int(combos.index(new))
                flow_out = k_ons[j]/epsilon**num_mismatch
                H[i][new_index] = flow_out
                flows_out += [flow_out]
                
            elif current[j] == 1:
                new[j] = 0
                # Find index of new string in basis.
                new_index = int(combos.index(new))
                flow_out = k_off/epsilon**num_mismatch
                H[i][new_index] = flow_out
                flows_out += [flow_out]
                
            else:
                print('problem')
        
        total_out = sum(flows_out)
        H[i][i] = -total_out
        
    return H

# # Example Ising matrices. 

# H_ising = np.array([[-k_on*(2), k_on, k_on, 0], 
#        [k_off/epsilon, -(k_on+k_off)/epsilon, 0, k_on/epsilon],
#       [k_off/epsilon, 0, -(k_on+k_off)/epsilon, k_on/epsilon],
#       [0, k_off, k_off, -k_off*(2)]])

# e = epsilon
# H_ising = np.array([[-k_on*(3), k_on, k_on, k_on, 0, 0, 0, 0], 
#          [k_off/e, -(k_on*2 + k_off)/e, 0, 0, k_on/e, k_on/e, 0, 0],
#          [k_off/e**2, 0, -(k_on*2 + k_off )/e**2, 0, k_on/e**2, 0, k_on/e**2, 0],
#          [k_off/e, 0 , 0, -(k_off + k_on*(2))/e, 0, k_on/e, k_on/e, 0 ],
#          [0, k_off/e, k_off/e, 0, -(k_on + 2*k_off)/e, 0, 0, k_on/e],
#          [0, k_off/e**2, 0, k_off/e**2, 0, -(k_on + k_off*2)/e**2, 0, k_on/e**2 ],
#          [0, 0, k_off/e, k_off/e, 0, 0, -(k_on + 2*k_off)/e, k_on/e],
#          [0, 0, 0, 0, k_off, k_off, k_off, -3*k_off]
#               ])

# Check ss prob distribution.
def transition_steady_state(H):
    '''
    Calculate steady state distribution given transition matrix, H.
    '''
    Hss = scipy.linalg.null_space(H.T)
    Hss /= Hss.sum()
    Hss = Hss.squeeze()
    
    return Hss

def Ising_steady_state(k_ons, k_off, epsilon, periodic=False):
    '''
    Calculate desired Ising steady state analytically.
    '''
    n = len(k_ons)
    N = 2**n
    
    combos = basis_sort(create_binaries(n))

    Hss_Ising = np.zeros(N)
    # Iterate through combinations and their transitions to other strings.
    for i in range(N):
        prob = 1
        
        string = combos[i]
        num_mismatch = count_mismatch(string, periodic=periodic)
        prob = prob*epsilon**num_mismatch

        for j in range(n):
            if string[j]==1:
                prob = prob*k_ons[j]
            else:
                 prob = prob*k_off
                    
        Hss_Ising[i] = prob
        
    Hss_Ising /= Hss_Ising.sum()
    return Hss_Ising

# Check that transition matrix null-space and Ising steady state align.
if __name__ == "__main__":
        
    # Define a custom argument type for a list of strings
    def list_of_floats(arg):
        out = [float(i) for i in arg.split(',')]
        return out
    
    parser = argparse.ArgumentParser(description='Create a transition matrix')
    parser.add_argument('--kons', type=list_of_floats, required=True,
                        help='List of k_on values')
    parser.add_argument('--koff', type=float, required=True,
                        help='Rate of switching off')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Inverse correlation strength')
    parser.add_argument('--periodic', type=bool, required=False,
                        help='Is locus periodic?')
    args = parser.parse_args()
    
    k_ons, k_off, epsilon, periodic = args.kons, args.koff, args.epsilon, args.periodic
    H = construct_H(k_ons, k_off, epsilon, periodic=periodic)
    Ising_ss = Ising_steady_state(k_ons, k_off, epsilon, periodic=periodic)
    Markov_ss = transition_steady_state(H)
    
    print('Ising steady-state distribution:')
    print(Ising_ss)
    print('Markov chain steady-state distribution')
    print(Markov_ss)
   