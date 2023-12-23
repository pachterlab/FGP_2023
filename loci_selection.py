#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy import matlib
import scipy.stats
from numpy.random import random

import numba
import tqdm
import multiprocessing
import scipy
from scipy.fft import ifft, ifft2, ifftn, irfft, irfftn
import pandas as pd

from scipy.stats import gaussian_kde as kde

# !pip --quiet install 'matplotlib==3.6.0'

import pkg_resources
pkg_resources.require("matplotlib==3.6.0")
import matplotlib

from matplotlib.colors import Normalize
from matplotlib import cm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import time
t1 = time.time()
import scanpy


# In[3]:


## Read-in count matrix, output of snATAK pipeline, cells_x_genes.mtx 
# Count matrix is in form cell:peak:count
out = 'output_merged'
gene_matrix_file = out+'/counts_mult/cells_x_genes.mtx'
matrix_dimension = pd.read_csv(gene_matrix_file, ' ' ,nrows=1, skiprows=3, skipinitialspace=True, header=None)
matrix_dimension = matrix_dimension.values[0]
cell_number, site_number, count_number = [int(i) for i in matrix_dimension[0:3]]
print(cell_number, 'Cells')
print(site_number, 'Sites')
print(count_number, 'Non-zero Matrix Entries')

# (Could find cell barcodes in cells_x_genes.barcodes.txt.)

# Create dataframe with ATAC data.
df_atac = pd.read_csv (gene_matrix_file, ' ', skiprows=4, skipinitialspace=True, header=None)
df_atac.columns = ['Cell', 'Gene', 'Count']

# Create dataframe mapping each transcript to a genomic location.
transcript_file = out + '/transcripts.txt'
df_transcripts = pd.read_csv (transcript_file, '\t', skiprows=0, skipinitialspace=True, header=None)
df_transcripts.columns = ['location']
df_transcripts.index.name = 'transcript'
df_transcripts.index = df_transcripts.index+1


# In[ ]:


## Perform QC and remove cells with too few reads.



# In[6]:


# 508251 Cells
# 64646 Sites
# 59917210 Non-zero Matrix Entries


# In[7]:


# old = 'ref_1_rev_com/transcripts.txt'
# new = 'ref_1_rev_com/counts_mult/cells_x_genes.genes.txt'
# ! cmp --silent $old $new || echo "files are different"


# In[6]:


# Find indices in transcripts where adjacent the following peak is less than 1kbp away.
def test_close(a,b,distance):
    '''
    Test whether peak-sites a and b are closer than a given distance.
    
    Args:
        a (string): first peak-site, in form chrom:pos1-pos2 (e.g. 1:10003-10460)
        b (string): second peak-site, in form chrom:pos1-pos2 (e.g. 1:10003-10460)
        distance (float): distance threshold in bp
        
    Output:
        bool: whether or not peaks a and b are separated by at most this distance.
    '''
    
    chromosome_a, position_a = a.split(':')
    chromosome_b, position_b = b.split(':')
    
    if chromosome_a != chromosome_b:
        return False
    
    a_start, a_end = int(position_a.split('-')[0]), int(position_a.split('-')[1])
    b_start, b_end = int(position_b.split('-')[0]), int(position_b.split('-')[1])
    
    if b_start - a_end <= distance:
        return True
    else:
        return False

## Test whether each ATAC site is close to its neighbor, according to our distance threshold in bp.
distance_threshold = 1500
df_transcripts['close']=0
for transcript in df_transcripts.index[:-1]:
    a= df_transcripts.loc[transcript]['location']
    b = df_transcripts.loc[transcript+1]['location']
    df_transcripts['close'][transcript] = test_close(a,b,distance_threshold)
df_transcripts['close'].iloc[-1] = 0
    
test_close('1:10003-10460', '1:11460-29373',1000)


# In[31]:


## Find continguous loci of at least length 6.
loci_dfs = []
new_locus = pd.DataFrame(columns=['transcript', 'location', 'close'])
n_locus = 0

for i in df_transcripts.index:
    if df_transcripts['close'].loc[i]:
        n_locus += 1
        row = df_transcripts.loc[i]
        new_locus.loc[len(new_locus.index)] = [i, row[0], row[1]]
    else:
        if n_locus>=4:
            loci_dfs += [new_locus]
        new_locus = pd.DataFrame(columns=['transcript', 'location', 'close'])
        n_locus = 0


# In[32]:


relevant_sites = []
for i in loci_dfs:
    if len(i) ==6:
        relevant_sites += list(i['transcript'])
print(relevant_sites)


# In[8]:


print(len(loci_dfs), 'loci found')
loci_lengths = [len(df) for df in loci_dfs]
print('lengths:', loci_lengths)
print(loci_lengths.count(6))
print(max(loci_lengths))


# In[39]:





# In[28]:


max(df_atac['Gene'])


# In[40]:


## QC to remove cells with low number of reads.
cell_totals  = df_atac[['Cell', 'Count']].groupby("Cell").sum()

# Define your series
s = pd.Series(cell_totals['Count'], name='Count')
knee_df = pd.DataFrame(s)

# Frequency
stats_df = knee_df \
.groupby('Count') \
['Count'] \
.agg('count') \
.pipe(pd.DataFrame) \
.rename(columns = {'Count': 'frequency'})

# PDF
stats_df['pdf'] = stats_df['frequency'] #/ sum(stats_df['frequency'])

# CDF
stats_df['cdf'] = stats_df['pdf'].cumsum()
stats_df['rank'] = stats_df['cdf'].iloc[-1] - stats_df['cdf']
stats_df = stats_df.reset_index()

## Create knee plot to identify UMI threshold.

stats_df.plot(x = 'rank', y = 'Count', grid = True, logx=True, logy=True)
plt.title('Knee Plot');
plt.xlabel('Rank')
plt.ylabel('Count')
plt.savefig('hi_' + out)


# In[41]:





# In[42]:


# Define UMI threshold at knee, and identify high quality cells.
UMI_threshold = 2000
high_cells = cell_totals[cell_totals['Count']>=threshold]

print('Estimated High-quality number of cells', len(high_cells))
print('Median Fragments overlapping peaks per Cell', high_cells['Count'].median())

# Create filtered count matrix with high quality cells.
QC_df_atac = df_atac[df_atac['Cell'].isin(high_cells.index)]


# In[43]:


print('Estimated High-quality number of cells', len(high_cells))
print('Median Fragments overlapping peaks per Cell', high_cells['Count'].median())


# In[44]:


# Filter based on count number, and retain info from relevant loci.
QC_relevant = QC_df_atac[QC_df_atac['Gene'].isin(relevant_sites)]
cell_num = len(list(set(QC_relevant['Cell'])))


# In[45]:


QC_relevant


# In[36]:


# Plot UMI numbers per cell
high_cells['num_human'] = 0
high_cells['num_mouse'] = 0
high_cells['organism'] = 0
high_cells['ratio'] = 0

# Assume transcripts are in order: human then mouse.
# Find first occurrence of mouse transcript.
get_ipython().system(" grep -m1 -n 'm' $transcript_file")
first_mouse_line = 58016
for cell in high_cells.index:
    
    hum_mask = (QC_df_atac['Cell']==cell) & (QC_df_atac['Gene']<first_mouse_line)
    hum_genes = QC_df_atac[hum_mask]
    num_human = sum(hum_genes['Count'].values)
    high_cells.loc[cell, 'num_human'] = num_human
    mouse_mask = (QC_df_atac['Cell']==cell) & (QC_df_atac['Gene']>=first_mouse_line)
    mouse_genes = QC_df_atac[mouse_mask]
    num_mouse = sum(mouse_genes['Count'].values)
    high_cells.loc[cell, 'num_mouse'] = num_mouse
    if num_mouse > num_human:
        high_cells.loc[cell, 'organism'] = 'mouse'
        ratio = num_mouse/num_human
        high_cells.loc[cell, 'ratio'] = ratio

    elif num_human > num_mouse:
        high_cells.loc[cell, 'organism'] = 'human'
        ratio = num_human/num_mouse
        high_cells.loc[cell, 'ratio'] = num_human/num_mouse

    else:
        high_cells.loc[cell, 'organism'] = 'ambiguous'
        
    # Assign value ambiguous to organism if ratio
    if ratio < 5:
        high_cells.loc[cell, 'organism'] = 'ambiguous'
    # print(genes)
    # num_human = len(genes.values[genes.values<first_mouse_line])
    # num_mouse = len(genes.values[genes.values>=first_mouse_line])
    


# In[37]:


high_cells


# In[38]:


# Flag doublets by checking ratio between organism counts is high enough.
high_cells['ratio'] = high_cells['num_mouse']/high_cells['num_human']
high_cells['organism'][(high_cells['ratio'] < 1)] = 'human'
high_cells['organism'][(high_cells['ratio'] > 1)] = 'mouse'

ratio_threshold = 10
high_cells['organism'][(high_cells['ratio'] < ratio_threshold) & (high_cells['ratio'] > 1/ratio_threshold)] = 'ambiguous'


# In[118]:


# Your data
col_dict = {'mouse': 'orange', 'human': 'blue', 'ambiguous': 'grey'}
colors = [col_dict[i] for i in high_cells['organism']]

# Separate the data into 'mouse' and 'human' and create scatter plots with labels
mouse_indices = high_cells['organism'] == 'mouse'
human_indices = high_cells['organism'] == 'human'
ambig_indices = high_cells['organism'] == 'ambiguous'


plt.scatter(high_cells['num_human'][mouse_indices], high_cells['num_mouse'][mouse_indices], c='orange', label='mouse')
plt.scatter(high_cells['num_human'][human_indices], high_cells['num_mouse'][human_indices], c='blue', label='human')
plt.scatter(high_cells['num_human'][ambig_indices], high_cells['num_mouse'][ambig_indices], c='grey', label='ambiguous')


plt.title('Mixed Mouse/Human ATAC Counts')
plt.xlabel('Human Peak Counts')
plt.ylabel('Mouse Peak Counts')
plt.xscale('log')
plt.yscale('log')

plt.legend() 
plt.savefig('peak_by_organism.png', bbox_inches='tight', dpi=300)


# In[119]:


## Save separate mouse, human csvs for all high quality cells, saving only relevant genes.

# Extract cell numbers from the index of high_cells
mouse_cell_numbers = high_cells[mouse_indices].index
human_cell_numbers = high_cells[human_indices].index

# Filter rows in QC_df_atac based on cell numbers
mouse_df = QC_relevant[QC_relevant['Cell'].isin(mouse_cell_numbers)]
human_df = QC_relevant[QC_relevant['Cell'].isin(human_cell_numbers)]

# Define the file paths for saving the DataFrames to CSV files
mouse_csv_file = 'mouse_cells_data.csv'
human_csv_file = 'human_cells_data.csv'

# Save the DataFrames to CSV files
mouse_df.to_csv(mouse_csv_file, index=False)  # Save the mouse cell data without index
human_df.to_csv(human_csv_file, index=False)  # Save the human cell data without index


# In[35]:


# Load the CSV files into DataFrames.
mouse_df = pd.read_csv('mouse_cells_data.csv')
human_df = pd.read_csv('human_cells_data.csv')

# Verify the loaded DataFrames
print("Mouse DataFrame:")
print(mouse_df.head())

print("\nHuman DataFrame:")
print(human_df.head())


# In[33]:


len6_loci = [i for i in loci_dfs if len(i)==6]
loci_transcripts = [i['transcript'].values for i in len6_loci]


# In[36]:


## Get human cell distribution.
# Get distribution from sparse matrix.
human_loci_dictionary = {tuple(key): {} for key in loci_transcripts}

# Create dictionary with which sites are open at which cells.
for i in range(int(len(human_df))):
    
    row = human_df.iloc[i]
    gene = row['Gene']
    cell = row['Cell']
    
    for locus in loci_transcripts:
        if gene in locus:
            locus_key = tuple(locus)
            if cell in human_loci_dictionary[locus_key].keys():
                human_loci_dictionary[locus_key][cell] += [gene]
            else:
                human_loci_dictionary[locus_key][cell] = [gene]
            break
        else:
            pass
                
# loci_dictionary[tuple(my_locus)] = cell_count_dict
    # print(loci_dictionary)


# In[74]:


# Get distribution from sparse matrix.
# loci_dictionary = dict.fromkeys([tuple(i) for i in loci_transcripts], {})
mouse_loci_dictionary = {tuple(key): {} for key in loci_transcripts}

# Create dictionary with which sites are open at which cells.
for i in range(int(len(mouse_df))):
    
    row = mouse_df.iloc[i]
    gene = row['Gene']
    cell = row['Cell']
    
    for locus in loci_transcripts:
        if gene in locus:
            locus_key = tuple(locus)
            if cell in mouse_loci_dictionary[locus_key].keys():
                mouse_loci_dictionary[locus_key][cell] += [gene]
            else:
                mouse_loci_dictionary[locus_key][cell] = [gene]
            break
        else:
            pass
                
# loci_dictionary[tuple(my_locus)] = cell_count_dict
    # print(loci_dictionary)


# In[8]:


# Get distribution from dictionary of cells with their open sites.

# Given a tuple locus and a list of sites, return the string associated with those sites.
def strings_from_sites(locus, sites):
    locus = list(locus)
    string = []
    for i in locus:
        if i in sites:
            string += [1]
        else:
            string += [0]
    return tuple(string)

def strings_from_cell_dict(locus, cell_dict):
    return [strings_from_sites(locus, site_list) for site_list in cell_dict.values()]

def dist_from_cell_dict(locus, cell_dict, cell_num):
    dist = {tuple(i):0 for i in combos}
    tot = 0
    for string in strings_from_cell_dict(locus, cell_dict):
        dist[string] += 1
        tot += 1
    dist[(0,0,0,0,0,0)] = cell_num - tot
    return dist

# sum(dist_from_cell_dict(k, loci_dictionary[k], cell_num).values())==cell_num


# In[17]:


# mouse_loci_dictionary


# In[ ]:


loci_dictionary


# In[46]:


loci_dictionary = human_loci_dictionary
# Get distribution for each locus.
dist_dict = {locus:dist_from_cell_dict(locus, loci_dictionary[locus], cell_num) for locus in loci_dictionary.keys()}


# In[47]:


# # Save human dictionary to file.

# with open('mixed_human_dist_dict.pkl', 'wb') as fp:
#     pickle.dump(dist_dict, fp)
#     print('dictionary saved successfully to file')


# In[128]:


# import pickle

# # save dictionary to person_data.pkl file
# with open('mixed_mouse_dist_dict.pkl', 'wb') as fp:
#     pickle.dump(dist_dict, fp)
#     print('dictionary saved successfully to file')


# In[9]:


# import pickle
# with open('mixed_mouse_dist_dict.pkl', 'rb') as fp:
#     dist_dict = pickle.load(fp)


# In[49]:


dist_dict.keys()

