import pandas as pd
# Source: https://signor.uniroma2.it/downloads.php
signor_file = 'SIGNOR_all_data_14_08_25.tsv'

signor_data = pd.read_csv(signor_file, index_col=None, sep='\t')
print(signor_data.columns)

new_rows = []

# subject_id  object_id   subject_id_prefix   object_id_prefix    subject_name    object_name predicate   Primary_Knowledge_Source    Knowledge_Source    publications    subject_category    object_category   score

for _, row in signor_data.iterrows():
    subject_id = row.IDA
    object_id = row.IDB
    subject_id_prefix = row.DATABASEA
    object_id_prefix = row.DATABASEB
    subject_name = row.ENTITYA
    object_name = row.ENTITYB
    predicate = row.EFFECT
    new_row = {'subject_id': subject_id,
            'object_id': object_id,
            'subject_id_prefix': subject_id_prefix,
            'object_id_prefix': object_id_prefix,
            'subject_name': subject_name,
            'object_name': object_name,
            'predicate': predicate,
            'Primary_Knowledge_Source': 'SIGNOR',
            'Knowledge_Source': 'SIGNOR',
            'publications': row.PMID,
            'subject_category': row.TYPEA,
            'object_category': row.TYPEB,
            'score': row.SCORE
            }
    new_rows.append(new_row)

new_table = pd.DataFrame(new_rows)
new_table.to_csv('SIGNOR_2025_08_14.tsv', index=False, sep='\t')
