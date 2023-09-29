import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import itertools
from tqdm import tqdm
import networkx as nx
import seaborn as sns
import shapely
from shapely.geometry import linestring
from scipy.stats import gaussian_kde




def plot_votes_per_congress(path='../dataset/download_votes_merged/*'):
    res = {}
    for csv in tqdm(glob.glob(path)):                                                               # iterate over all csv files in the directory
        if '.csv' in csv:                                                                           # check if it is a csv file
            congress = int(os.path.basename(csv).replace('congress_','').replace('.csv',''))        # extract the congress number from the file name
            res[congress] = len(set(pd.read_csv(csv)['id']))                                        # count the number of unique votes in the file and store it in a dictionary

    sort_ = sorted(res.items())                                                                     # sort the dictionary by key
    x, y = zip(*sort_)                                                                              # unzip the dictionary into two lists
    
    plt.plot(x, y)

    plt.suptitle('Number of votes per congress', fontsize=20)
    plt.xlabel('congress number', fontsize=18)
    plt.ylabel('number of votes', fontsize=16)

    plt.show()
    
    return res



def plot_members_per_congress(path='../dataset/download_votes_merged/*'):
	res = {}
	for csv in tqdm(glob.glob(path)):
		if '.csv' in csv:
			congress = int(os.path.basename(csv).replace('congress_','').replace('.csv',''))
			res[congress] = len(set(pd.read_csv(csv)['name']))

	sort_ = sorted(res.items())
	x, y = zip(*sort_)
	
	plt.plot(x, y)

	plt.suptitle('Number of members per congress', fontsize=20)
	plt.xlabel('congress number', fontsize=18)
	plt.ylabel('number of members', fontsize=16)
	# start the y axis from 0
	plt.ylim(bottom=0, top=max(y)+100)


	plt.show()
	
	return res



def create_members_df(members, party_codes):
    temp_congress = members.groupby('icpsr', as_index=False)[['congress']].agg(lambda x: list(x))                                                           # group by icpsr and aggregate the congress numbers into a list
    temp_party = members.groupby('icpsr', as_index=False)[['party_code']].agg(lambda x: list(set(x)))                                                       # group by icpsr and aggregate the party codes into a list
    temp_congress = temp_congress.merge(temp_party)                                                                                                         # merge the two dataframes
    temp_congress['bioname'] = temp_congress['icpsr'].map(members[['icpsr', 'bioname']].set_index('icpsr').to_dict()['bioname'])                            # insert the bioname based on the icpsr 
    temp_congress['state_abbrev'] = temp_congress['icpsr'].map(members[['icpsr', 'state_abbrev']].set_index('icpsr').to_dict()['state_abbrev'])             # insert the state_abbrev based on the icpsr
    party_codes_dic = party_codes[['party_name', 'party_code']].set_index('party_code').to_dict()['party_name']                                             # create a dictionary for the party codes
    temp_congress['party_name'] = temp_congress['party_code'].apply(lambda x: [party_codes_dic[y] for y in x])                                              # insert the party name based on the party code
    return temp_congress



def edgelist_from_congress(congress, members_party_dict):
	edgelist = pd.DataFrame()

	for voteid in tqdm(set(congress['id'])):                         # iterate over all votes id (ids are unique for each vote)
    
		temp = congress[congress['id'] == voteid]                 # select the rows where the vote id is equal to the current vote id            
    
		yy = temp[temp['vote']=='Yea']['icpsr']                         # select the icpsr of the members that voted "Yea"
		nn = temp[temp['vote']=='Nay']['icpsr']                         
    
		y = itertools.combinations(yy, 2)                # all possible combinations of 2 members that voted "Yea"
		n = itertools.combinations(nn, 2)                
		o = itertools.product(yy, nn)                    # cartesian product of the 2 series

		y = pd.DataFrame(y, columns = ['source', 'target'])     # create a dataframe from the combinations of "Yea" voters
		y['weight'] = 1                                         # add a column with the weight of the edge
		y['count'] = 1                                         
    
		n = pd.DataFrame(n, columns = ['source', 'target'])     
		n['weight'] = 1                                         
		n['count'] = 1                                          
    
		o = pd.DataFrame(o, columns = ['source', 'target'])     
		o['weight'] = -1                                    # same but the link is negative                    
		o['count'] = 1                                          

		edgelist = pd.concat([edgelist, y, n, o])                    


	edgelist = pd.concat([edgelist, pd.DataFrame({
		'source': edgelist['target'],                   # new columns based on old columns: 
		'target': edgelist['source'],                   #   'newcolumn': dataframe['oldcolumn']
		'weight': edgelist['weight'],
		'count': edgelist['count']})])


	edgelist = edgelist.loc[edgelist['source'] < edgelist['target']]                    # remove duplicates
	edgelist = edgelist.groupby(['source', 'target', 'weight']).sum().reset_index()     # group by source, target and weight and sum the count
	edgelist['party'] = edgelist.apply(lambda row: 'in' if members_party_dict[row['source']] == members_party_dict[row['target']] else 'out', axis=1)   # create a column with the party of the edge

	map_votes = edgelist.groupby(['source', 'target'])['count'].sum().to_dict()                                                                         # create a dictionary with the number of votes togheter for each pair of nodes                               

	edgelist['votes_togheter'] = edgelist[['source', 'target']].apply(lambda x: map_votes[(x['source'], x['target'])], axis=1)
	edgelist['perc'] = edgelist['count']/edgelist['votes_togheter']
     
	return edgelist
     


def plot_kde(df, weight):
    

	def _midpoint(p1, p2):
		return {'x': (p1['x']+p2['x'])/2, 'y': (p1['y']+p2['y'])/2}

	def line_intersection(in_party, out_party, intersect_points):
		index_in = np.argmax(in_party[1])
		index_out = np.argmax(out_party[1])

        # points of the mean of the distributions 
		point_in={'x': in_party[0][index_in], 'y': in_party[1][index_in]}
		point_out={'x': out_party[0][index_out], 'y': out_party[1][index_out]}

        # medianpoint (mean of the means) of the two distributions
		midpoint = _midpoint(point_in, point_out)
        
        #find index of intersection closer to midpoint
		index_closer = np.argmin([np.sqrt( (p[0] - midpoint['x'])**2 + (p[1] - midpoint['y'])**2 ) for p in intersect_points])

        # return x value of closer intersection
		return intersect_points[index_closer][0]
    

	#label = "agree" if weight == 1 else "disagree"

	x0 = df.loc[(df['party']=='in')&(df['weight'] == weight)]['perc']
	x1 = df.loc[(df['party']=='out')&(df['weight'] == weight)]['perc']

	bw = len(x0)**(-1./(2+4))
	kde0 = gaussian_kde(x0, bw_method=bw)
	bw = len(x1)**(-1./(2+4))
	kde1 = gaussian_kde(x1, bw_method=bw)

	xmin = min(x0.min(), x1.min())
	xmax = max(x0.max(), x1.max())
	dx = 0.2 * (xmax - xmin) # add a 20% margin, as the kde is wider than the data
	xmin -= dx
	xmax += dx

	x = np.linspace(xmin, xmax, 500)
	kde0_x = kde0(x)
	kde1_x = kde1(x)
	inters_x = np.minimum(kde0_x, kde1_x)

	idx = np.argwhere(np.diff(np.sign(kde0_x - kde1_x))).flatten()

	threshold = line_intersection([x, kde0_x], [x, kde0_x], [[x,y] for x,y in zip (x[idx], kde1_x[idx])])

	fig1, ax1 = plt.subplots(1, 1, figsize=(15, 10))
	fig1.set_size_inches(14, 10)

	ax1.plot(x, kde0_x, color='b', label='intra-party')
	ax1.fill_between(x, kde0_x, 0, color='b', alpha=0.2)

	ax1.plot(x, kde1_x, color='orange', label='inter-party')
	ax1.fill_between(x, kde1_x, 0, color='orange', alpha=0.2)

	ax1.plot(x, inters_x, color='tomato')
	ax1.fill_between(x, inters_x, 0, facecolor='none', edgecolor='tomato', label='intersection', alpha=0.5, hatch='xx')

	ax1.axvspan(threshold-0.005, threshold+0.005,color='tomato', alpha=0.7, zorder=10)
	ax1.text(threshold-.25, .93, 'threshold: '+str(round(threshold, 2)), fontsize=23,transform=ax1.transAxes)

	area_inters_x = np.trapz(inters_x, x)

	handles, labels = plt.gca().get_legend_handles_labels()
	labels[2] += f': {area_inters_x * 100:.1f} %'
	ax1.tick_params(axis='both', which='major', labelsize=20)

	plt.xlabel('Edges percentage', fontsize=25)
	plt.ylabel('Density', fontsize=25)
	title = "Positive edges" if weight == 1 else "Negative edges"
	c_title = "g" if weight == 1 else "r"
	plt.title(title, fontsize=31, pad=10, ha='left', x=-.1, c=c_title)

	legend1 = plt.legend([handles[0],handles[1]], [labels[0],labels[1]], loc='upper center', bbox_to_anchor=(0.4, 1.08), frameon=False, ncol=2, fontsize=23)
	plt.legend([handles[2]], [labels[2]], loc='upper center', bbox_to_anchor=(0.84, 1.08), frameon=False, ncol=1, fontsize=23)
	plt.gca().add_artist(legend1)
	plt.tight_layout()
	plt.grid(axis='y')
    
	ax1.set_xlim([-0.07, 1.1])
    
	plt.show()
	return threshold, area_inters_x