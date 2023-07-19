#!/usr/bin/env python
# coding: utf-8

# Note: This program takes awhile to run. Always click "wait" if it ascks you if you want to terminate the program.
# The graphs take awhile to load and close, however, the program will run and successfully complete.

import pandas as pd
import gffpandas.gffpandas as gffpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram


snp_df = pd.read_csv("../data/SNP_variant_data.csv")
bg_tpm_df = pd.read_csv("../data/bg_tpm_df.csv")
amyg_tpm_df = pd.read_csv("../data/amyg_tpm_df.csv")
fc_tpm_df = pd.read_csv("../data/fc_tpm_df.csv")
skel_tpm_df = pd.read_csv("../data/skeletal_muscle.csv")
#annotations = gffpd.read_gff3('../data/GRCh38_latest_genomic.gff')



# In[27]:


snp_df['Variant_Risk_Allele'] = snp_df['Variant and risk allele'].str.split('-').str[0]
snp_df = snp_df.drop(columns="Variant and risk allele", axis =1)
snp_df = snp_df.reindex(columns=['Variant_Risk_Allele', 'Location', 'Mapped gene', 'Trait(s)', 'Reported trait', 'Study accession'])
snp_df.drop(snp_df.loc[snp_df['Location']=="Mapping not available"].index, inplace=True)
snp_df['chromosome'] = snp_df.apply(lambda row: row['Location'].split(":")[0], axis=1)
snp_df['position'] = snp_df.apply(lambda row: row['Location'].split(":")[1], axis=1)


# In[28]:


# best_ref_df = annotations.df.loc[annotations.df['source'] == "BestRefSeq"]


# In[29]:


# gene_exon_df = best_ref_df[best_ref_df['attributes'].astype(str).str.contains("ID=gene|ID=exon")]


# In[30]:


# pd.options.mode.chained_assignment = None
# gene_exon_df['attributes_id'] = gene_exon_df.apply(lambda row: row['attributes'].split(";")[0].split("-")[0][3:], axis=1)
# pd.options.mode.chained_assignment = 'warn'


# In[31]:


# gene_exon_df[gene_exon_df['attributes'] == "ID=gene-ATP5MF-PTCD1;Dbxref=GeneID:100526740,HGNC:HGNC:38844;Name=ATP5MF-PTCD1;description=ATP5MF-PTCD1 readthrough;gbkey=Gene;gene=ATP5MF-PTCD1;gene_biotype=protein_coding;gene_synonym=ATP5J2-PTCD1"]


# In[32]:


dfs = {
    "bg_df": bg_tpm_df,
    "fc_df": fc_tpm_df,
    "amyg_df": amyg_tpm_df,
    "skel_df": skel_tpm_df
}
cluster_mapping = {
    "bg_df": {},
    "fc_df": {},
    "amyg_df": {},
    "skel_df": {}
}


# In[33]:


# Add Variance

for df in dfs.keys():
    important_cols_df = dfs[df].drop(['id', 'Name', 'Description'], axis=1)
    dfs[df]['variance'] = important_cols_df.var(axis=1)


# In[34]:


# Trim tpm_df to high variance rows
trim_value = 10000

for df in dfs.keys():
    dfs[df] = dfs[df].sort_values(by='variance', ascending=False)
    dfs[df] = dfs[df].head(trim_value)
    dfs[df] = dfs[df].reset_index(drop=True)


# In[35]:


dfs['bg_df']


# In[36]:


###############################
#   Hierarchical clustering   #
###############################

num_clusters = 200

for df in dfs.keys():  
    cluster_df = dfs[df]
    dropped_cols = cluster_df[['id', 'Name', 'Description', 'variance']]
    cluster_df = cluster_df.drop(['id', 'Name', 'Description', 'variance'], axis=1)

    #distance_matrix = linkage(cluster_df, method = 'ward', metric = 'euclidean')
    #distance_matrix = linkage(cluster_df, method = 'complete', metric = 'correlation')
    distance_matrix = linkage(cluster_df, method = 'complete', metric = 'correlation')
    dn = dendrogram(distance_matrix)
    # Display the dendogram
    title = ""
    if df == "bg_df":
        title = "Basal Ganglia"
    elif df == "fc_df":
        title = "Frontal Cortex"
    elif df == "amyg_df":
        title = "Amygdala"
    elif df == "skel_df":
        title = "Skeletal Muscle"
    else:
        print("DF Unknown")
    plt.title(title)
    plt.show()
    
    cluster_df['cluster_labels'] = fcluster(distance_matrix, num_clusters, criterion='maxclust')
    dfs[df] = pd.concat([dropped_cols, cluster_df], axis=1)
    





# In[37]:


# https://predictivehacks.com/hierarchical-clustering-in-python/


# In[ ]:





# In[ ]:





# In[38]:


# Gathers genes related to "alcohol dependence"
mask = snp_df['Trait(s)'].str.contains("alcohol dependence")
#mask = snp_df['Trait(s)'].str.contains("opioid dependence")
#mask = snp_df['Trait(s)'].str.contains("nicotine dependence")
interested_df = snp_df[mask]
interested_genes = interested_df['Mapped gene']
interested_genes_filtered = []
for gene in interested_genes:
    genes = gene.split(",")
    for elm in genes:
        if elm != "'-":
            interested_genes_filtered.append(elm.replace(" ", ""))
interested_genes_filtered = list(set(interested_genes_filtered))


# In[39]:


snp_df['Trait(s)'].unique()


# In[40]:


cluster_data = {
    "bg_df": [([],i+1,[]) for i in range (num_clusters)],
    "fc_df": [([],i+1,[]) for i in range (num_clusters)],
    "amyg_df": [([],i+1,[]) for i in range (num_clusters)],
    "skel_df": [([],i+1,[]) for i in range (num_clusters)]
}

# All of the below is just for the amyg tissue data
for df in dfs.keys():
    cluster_counts = [0] * num_clusters
    count = 0

    for i in range (len (dfs[df])):
        clust_lab = dfs[df]['cluster_labels'][i]    
        cluster_counts [clust_lab-1] += 1
        cluster_data[df][clust_lab-1][0].append(dfs[df]['Description'][i])
        cluster_data[df][clust_lab-1][2].append(dfs[df]['Name'][i])
    #print(dfs[df]['Name'])
    
    print (f"Cluster count ({df}):", num_clusters)
    print (f"Clusters ({df}):", cluster_counts)
    print (f"Mean cluster pop ({df}):", sum (cluster_counts) / len (cluster_counts))
    print ("===================================================================\n")


# In[ ]:





# In[41]:


print("############################")
print("#     Alcohol Dependence   #")
print("############################\n\n")

print ("\nNumber of interesting genes in total:", len(interested_genes_filtered), "\n\n\n")
interesting_genes_set = set (interested_genes_filtered)
    
for df in dfs.keys():
    intersects = []
    max_cluster = {
        "idx":-1,
        "counts": -1
    }
    for (desc, lab, name) in cluster_data[df]:
        inter = list (interesting_genes_set.intersection (desc))
        intersects.append ((inter, lab, len(desc)))
        if len(desc) > max_cluster["counts"]:
            max_cluster['idx'] = lab
            max_cluster['counts'] = len(desc)        

        
    intersects.sort(key = lambda pair: len(pair[0]), reverse = True)
    # each element of `intersects` is a pair `(inter_genes,lab)` where `inter_genes` is all of the genes 
    # from cluster `lab` that were in the `interesting_genes_filtered` list
    # `intersects[0]` represents the cluster with the highest intersection between its genes and the interesting ones
    
    title = ""
    if df == "bg_df":
        title = "Basal Ganglia"
    elif df == "fc_df":
        title = "Frontal Cortex"
    elif df == "amyg_df":
        title = "Amygdala"
    elif df == "skel_df":
        title = "Skeletal Muscle"
    else:
        print("DF Unknown")
    print(f"=================================  {title}  ========================================")
    print(f"Cluster {intersects[0][1]} had the most intersections with the interesting genes set.\n")
    print(f"The intersecting genes are: \n{intersects[0][0]}\n")
    print(f"Cluster {intersects[0][1]} had {intersects[0][2]} genes in the cluster.\n")
    print(f"Highest numbers of interesting genes in the top 5 clusters ({df}):", [len(inter[0]) for inter in intersects[:5]])
    print(f"MAX Cluster {max_cluster['idx']} with {max_cluster['counts']} genes.")
    print("\n")


# In[42]:


data = []
for df in dfs.keys():
    intersects = []
    for (desc, lab, name) in cluster_data[df]:
        inter = list (interesting_genes_set.intersection (desc))
        intersects.append ((inter, lab, len(desc), desc))
        if len(desc) > max_cluster["counts"]:
            max_cluster['idx'] = lab
            max_cluster['counts'] = len(desc)
    intersects.sort(key = lambda pair: len(pair[0]), reverse = True)

    
    title = ""
    if df == "bg_df":
        title = "Basal Ganglia"
    elif df == "fc_df":
        title = "Frontal Cortex"
    elif df == "amyg_df":
        title = "Amygdala"
    elif df == "skel_df":
        title = "Skeletal Muscle"
    else:
        print("DF Unknown")
    print(f"=================================  {title}  ========================================")
    data.append(intersects[0][3])
    for gene in intersects[0][3]:
        #print(gene)
        continue
    print("\n")
    
    # bar graphs - code ripped from https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/#
    
    keep = list (filter (lambda inter: len(inter[0])>0, intersects))
    
    totals = [inter[2] for inter in keep]
    interestings = [len(inter[0]) for inter in keep]
    labs = [str(inter[1]) for inter in keep]
    
    X_axis = np.arange(len(labs))
  
    #plt.bar(X_axis - 0.2, totals, 0.4, label = 'Total')
    plt.bar(X_axis - 0.6, interestings, label = 'Interesting')

    plt.xticks(X_axis, labs, rotation=90, ha='right')
    plt.xlabel("Cluster IDs")
    plt.ylabel("Number of genes")
    plt.title("Number interesting genes in clusters for " + title)
    #plt.legend()
    plt.show()
    plt.clf()
    


# In[43]:


common_elements = set(data[0]).intersection(data[1], data[2])
print(list(common_elements))


# In[44]:


list(dfs.keys())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




