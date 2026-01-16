# extremely simple way of converting gene names
import gzip
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'KG_files')

ID_TO_SYMBOL = {}
SYMBOL_TO_ID = {}
ID_TO_UNIPROT = {}
UNIPROT_TO_ID = {}
ID_TO_ENSEMBL = {}
ENSEMBL_TO_ID = {}


def _load_gene_info():
    with gzip.open(os.path.join(file_path, 'Homo_sapiens.gene_info.gz'), 'rt') as f:
        for row in f.readlines():
            row = row.split('\t')
            if row[0] == '#tax_id':
                continue
            gene_id = int(row[1])
            ID_TO_SYMBOL[gene_id] = row[2]
            synonyms = row[4].split('|')
            if row[2] not in SYMBOL_TO_ID:
                SYMBOL_TO_ID[row[2]] = gene_id
            for s in synonyms:
                if s not in SYMBOL_TO_ID:
                    SYMBOL_TO_ID[s] = gene_id

def _load_uniprot_info():
    with open(os.path.join(file_path, 'gene_uniprot.txt')) as f:
        for row in f.readlines():
            row = row.split()
            gene_id = int(row[0])
            if gene_id not in ID_TO_UNIPROT:
                ID_TO_UNIPROT[gene_id] = row[1]
            if row[1] not in UNIPROT_TO_ID:
                UNIPROT_TO_ID[row[1]] = gene_id
    with open(os.path.join(file_path, 'uniprot_genes.txt')) as f:
        for row in f.readlines():
            row = row.strip().split()
            gene_id = int(row[1])
            if row[0] not in UNIPROT_TO_ID:
                UNIPROT_TO_ID[row[0]] = gene_id

def _load_ensembl_info():
    with open(os.path.join(file_path, 'geneid_ensembl.txt')) as f:
        for row in f.readlines():
            row = row.split()
            gene_id = int(row[0])
            ID_TO_ENSEMBL[gene_id] = row[1]
            ENSEMBL_TO_ID[row[1]] = gene_id

_load_gene_info()
    
def convert(source, dest, ids, ignore_missing=True):
    """
    General conversion function.

    source/dest can be 'ncbi'/'geneid', 'symbol', 'uniprot', 'ensembl'
    """
    gene_ids = []
    if source == 'ncbi' or source == 'geneid':
        gene_ids = ids
    elif source == 'symbol':
        gene_ids = get_ids(ids, ignore_missing=ignore_missing)
    elif source == 'uniprot':
        gene_ids = uniprot_to_gene_ids(ids, ignore_missing=ignore_missing)
    elif source == 'ensembl':
        gene_ids = ensembl_to_gene_ids(ids, ignore_missing=ignore_missing)
    if dest == 'ncbi' or dest == 'geneid':
        return gene_ids
    elif dest == 'symbol':
        return get_symbols(gene_ids, ignore_missing=ignore_missing)
    elif dest == 'uniprot':
        return gene_ids_to_uniprot(gene_ids, ignore_missing=ignore_missing)
    elif dest == 'ensembl':
        return gene_ids_to_ensembl(gene_ids, ignore_missing=ignore_missing)

def get_symbols(gene_ids, ignore_missing=True):
    """
    Get symbols given a list of gene ids.
    """
    gene_ids = [int(x) for x in gene_ids]
    if ignore_missing:
        return [ID_TO_SYMBOL[x] for x in gene_ids if x in ID_TO_SYMBOL]
    return [ID_TO_SYMBOL[x] for x in gene_ids]

def get_symbol(gene_id):
    return ID_TO_SYMBOL[int(gene_id)]

def get_ids(gene_symbols, ignore_missing=True):
    """
    Get ids given a list of gene symbols.
    """
    if ignore_missing:
        return [SYMBOL_TO_ID[x] for x in gene_symbols if x in SYMBOL_TO_ID]
    return [SYMBOL_TO_ID[x] for x in gene_symbols]

def get_id(gene_symbol):
    return SYMBOL_TO_ID[gene_symbol]


def gene_ids_to_uniprot(gene_ids, ignore_missing=True):
    if len(ID_TO_UNIPROT) == 0:
        _load_uniprot_info()
    gene_ids = [int(x) for x in gene_ids]
    if ignore_missing:
        return [ID_TO_UNIPROT[x] for x in gene_ids if x in ID_TO_UNIPROT]
    return [ID_TO_UNIPROT[x] for x in gene_ids]


def uniprot_to_gene_ids(uniprot_ids, ignore_missing=True):
    if len(ID_TO_UNIPROT) == 0:
        _load_uniprot_info()
    if ignore_missing:
        return [UNIPROT_TO_ID[x] for x in uniprot_ids if x in UNIPROT_TO_ID]
    return [UNIPROT_TO_ID[x] for x in uniprot_ids]



def gene_ids_to_ensembl(gene_ids, ignore_missing=True):
    if len(ID_TO_ENSEMBL) == 0:
        _load_ensembl_info()
    gene_ids = [int(x) for x in gene_ids]
    if ignore_missing:
        return [ID_TO_ENSEMBL[x] for x in gene_ids if x in ID_TO_ENSEMBL]
    return [ID_TO_ENSEMBL[x] for x in gene_ids]


def ensembl_to_gene_ids(ensembl_ids, ignore_missing=True):
    if len(ID_TO_ENSEMBL) == 0:
        _load_ensembl_info()
    if ignore_missing:
        return [ENSEMBL_TO_ID[x] for x in ensembl_ids if x in ENSEMBL_TO_ID]
    return [ENSEMBL_TO_ID[x] for x in ensembl_ids]
