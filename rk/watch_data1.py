import pandas as pd
from os import rename
from datetime.datetime import now

class Watch:
    table = None
    def_name = '~/watch/last.csv'
    def_dir = '~/watch/'
    def __init__(self):
        table = pd.read_csv(def_name, index_col='index')

    def name_backup():
        time = now()
        return 'back%d.%d_%d:%d:%d.csv' % (now.month, now.day, now.hour, now.minute, now.second)

    def __del__(self):
        rename(def_name, join(def_dir, name_backup()))
        table.to_csv(def_name)
