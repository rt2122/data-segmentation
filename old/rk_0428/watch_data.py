import pandas as pd
from os import walk
from os.path import join
from all_p import ClusterFile



table = pd.DataFrame([])
table.index.name = 'index'

cf = ClusterFile('tcen_n50_il0_in4.csv')
p_cen = pd.read_csv('~/data/train/centers/tcen_n50_il0_in4.csv', index_col='Unnamed: 0')
print(list(p_cen))

table['id_patch'] = None
table['id_list'] = None
table['ra'] = None
table['dec'] = None
table['inpix'] = None
table['state'] = None

for i in range(p_cen.shape[0]):
    j = table.shape[0]
    table['id_patch'].loc[j] = i
    table['id_list'].loc[j] = 0
    table['ra'] = p_cen['ra']
    table['dec'] = p_cen['dec']
    table['inpix'] = 4

table.to_csv('~/watch/last.csv')
table.head()
