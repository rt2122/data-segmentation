import re
import pandas as pd

class ClusterFile:

    pnames = pd.DataFrame({'var':     ['typ', 'id_patch', 'ra', 'dec', 'state' , 'num',  'inpix', 'id_list'],
                           'short':   ['t',   'ip',       'ra', 'dec', 's',      'n',    'in',    'il'],
                           'val_type':['s',   'i',        'f',  'f',   's',      'i',    'i',     'i']})

    def __init__(self, name):
        self.params = {'typ' : None,
                'id_patch' : None,
                'ra' : None,
                'dec' : None,
                'state' : None,
                'num' : None,
                'inpix' : None,
                'id_list' : None}
        name = name[:-len('.csv')]
        name = re.split('_', name)
        for n in name:
            for i in range(self.pnames.shape[0]):
                sh_var = self.pnames['short'].iloc[i]
                if n.startswith(sh_var):
                    val = n[len(sh_var):]
                    if self.pnames['val_type'].iloc[i] == 'i':
                        val = int(val)
                    elif self.pnames['val_type'].iloc[i] == 'f':
                        val = float(val)
                    self.params[self.pnames['var'].iloc[i]] = val

    def file(self):
        s = ''
        for p in self.params:
            if not (self.params[p] is None):
                idx = self.pnames[self.pnames['var'] == p].index[0]
                sh = self.pnames['short'].iloc[idx]
                s += sh
                val = self.params[p]
                if re.match('float', str(type(val))):
                    s += '%.4f' % val
                else:
                    s += str(val)
                s += '_'
        s = s[:-1]
        return s + '.csv'



f = 'tdat_ip18_ra92.5061_dec85.5084_sne_in4_il0.csv'
cl = ClusterFile(f)
print(f)
print(cl.file())
