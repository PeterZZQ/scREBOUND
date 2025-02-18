# In[]
# ----------------------------------------------------------------------------
#
# Process the labels
#
# ----------------------------------------------------------------------------
import pandas as pd
from ontobio.ontol_factory import OntologyFactory
import torch
import numpy as np


# def get_ancestors(ont, cl_id):
#     """Return all ancestors of specified node.

#     The default implementation is to use networkx, but some
#     implementations of the Ontology class may use a database or
#     service backed implementation, for large graphs.

#     Arguments
#     ---------
#     ont: ontology class
#     node : str
#         identifier for node in ontology

#     Returns
#     -------
#     list[str]
#         ancestor node IDs

#     """
#     print(f"get ancester for {cl_id}...")
#     ancestor_list = []
#     parents = [cl_id]
#     # issue: the length will not be 0 sometimes
#     while len(parents) > 0:
#         parents_new = []
#         for parent_clid in parents:
#             parents_new.extend([x for x in ont.parents(parent_clid, relations = None) if x[:2] == "CL"])
        
#         # parents = [x for x in parents_new if x in ancestor_pool]
#         parents = [x for x in np.unique(parents_new)]
#         ancestor_list.extend(parents)

#     # issue: the ordering, cannot do unique
#     ancestor_list = [x for x in np.unique(ancestor_list)]

#     print("Done.")
#     return ancestor_list

def get_ancestors(ont, cl_id, remove_self = True):
    """Return all ancestors of specified node.

    The default implementation is to use networkx, but some
    implementations of the Ontology class may use a database or
    service backed implementation, for large graphs.

    Arguments
    ---------
    ont: ontology class
    node : str
        identifier for node in ontology

    Returns
    -------
    list[str]
        ancestor node IDs

    """
    seen = []
    nextnodes = [cl_id]
    while len(nextnodes) > 0:
        nn = nextnodes.pop()
        if not nn in seen:
            seen.append(nn)
            # NOTE: only include the CL ancestors
            nextnodes += [x for x in ont.parents(nn, relations = None) if x[:2] == "CL"]
    # remove the first item (query cl_id)
    if remove_self:
        seen = [x for x in seen[1:] if x[:2] == "CL"]
    else:
        seen = [x for x in seen if x[:2] == "CL"]
    seen = np.unique(seen)
    return seen

# In[]
n_mgene = 256
# read in the label of dataset
# data_dir = "/localscratch/ziqi/localscratch_tempdata/cellxgene/"
data_dir = "/project/zzhang834/LLM_KD/dataset/cellxgene/"

meta_dict = torch.load(data_dir + f"meta_{n_mgene}_pancreas.pt", weights_only = False)
# create the cell ontology tree from the labels
ont = OntologyFactory().create("/project/zzhang834/LLM_KD/dataset/cl.json")
label_ids, label_id_counts = np.unique(meta_dict["label"], return_counts = True)
label_ids = label_ids[np.argsort(label_id_counts)]
label_ids_code = meta_dict["label_code"][label_ids]
label_code_clid = np.array([x.split("--")[0] for x in label_ids_code])
label_id_counts = np.sort(label_id_counts)

# create the dataframe
label_code_df = pd.DataFrame(index = label_code_clid, columns = ["cell_type"], data = np.array([x.split("--")[1] for x in label_ids_code])[:,None])
label_code_df["counts"] = label_id_counts

# In[]
# Hierarchical labels
# NOTE: Method 1: Find the highest level (lowest granularity) labels, 
# there can be multiple ancestral cell types for each cell type, which cause ambiguity for high-level annotations

# label_code_df["root_clid"] = label_code_df.index.values
# label_code_df["root_celltype"] = label_code_df["cell_type"].values
# for idx, clid in enumerate(label_code_df.index.values.squeeze()):
#     ancesters = get_ancestors(ont, clid)
#     # find the overlap
#     ancesters = np.intersect1d(np.array([x for x in label_code_df.index.values]), ancesters)

#     root_ancesters = []
#     for ancester in ancesters:
#         # check if the ancesters of the ancester exist in the list
#         temp_ancesters = get_ancestors(ont, ancester)
#         if len(np.intersect1d(ancesters, temp_ancesters)) == 0:
#             root_ancesters.append(ancester)

#     if len(root_ancesters) > 1:
#         raise ValueError("More than one root ancesters...")

# # NOTE: root cell type annotation need to match the label_code order for the ease of reference 
# label_code_root = label_code_df.loc[np.array([x.split("--")[0] for x in meta_dict["label_code"]]), "root_celltype"].values
# label_root = np.array([label_code_root[x] for x in meta_dict["label"]])
# # redo the factorization
# label_root_id, label_root_code = pd.factorize(label_root)
# meta_dict["label_root"] = label_root_id
# meta_dict["label_code_root"] = label_root_code


# In[]
# NOTE: Method 2: use bincode to represent cell types, encode both the cell type and its ancestral cell types
# bincode is simple to use for classifier, multi-head classifier, but how to use it for contrastive loss??

label_bincode = pd.DataFrame(np.zeros((label_code_df.shape[0], label_code_df.shape[0]), dtype = int), index = label_code_df.index.values, columns = label_code_df.index.values)

for idx, clid in enumerate(label_code_df.index.values.squeeze()):
    # find all ancester ct for clid
    ancesters = get_ancestors(ont, clid)
    # find the overlapping clid and ancester cells
    ancesters = np.intersect1d(label_code_df.index.values.squeeze(), ancesters)
    # assign cell type
    label_bincode.loc[clid, clid] = 1
    # assign ancestral cell type
    label_bincode.loc[clid, ancesters] = 1


# NOTE: bincode need to match the label_code order for the ease of reference
label_code_bincode = label_bincode.loc[np.array([x.split("--")[0] for x in meta_dict["label_code"]]), :].values
meta_dict["label_bincode"] = label_code_bincode

torch.save(meta_dict, data_dir + f"meta_{n_mgene}_pancreas_rootannot.pt")


# %%
