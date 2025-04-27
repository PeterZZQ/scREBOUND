
from ontobio.ontol_factory import OntologyFactory
import torch
import numpy as np
import pandas as pd

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

def clid2bincode(clids, clid_pool = None):
    ont = OntologyFactory().create("/project/zzhang834/LLM_KD/dataset/cl.json")
    if clid_pool is None:
        clid_pool = np.unique(clids)
    label_bincode = pd.DataFrame(np.zeros((len(clids), len(clid_pool)), dtype = int),
                                 index = clids, columns = clid_pool)
    
    for clid in clids:
        # find all ancester ct for clid
        ancesters = get_ancestors(ont, clid)
        # find the overlapping clid and ancester cells
        ancesters = np.intersect1d(clid_pool, ancesters)
        # assign cell type
        label_bincode.loc[clid, clid] = 1
        # assign ancestral cell type
        label_bincode.loc[clid, ancesters] = 1
    
    return label_bincode
