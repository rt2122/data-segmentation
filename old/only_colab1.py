import pandas as pd
import mastcasjobs

def get_patch(ra, dec, radius, job_name='job', file_name='patch.csv', table_name='table'):
    wsid = 1540641525
    pwd = "emptystreets1"
    jobs = mastcasjobs.MastCasJobs(userid=wsid, password=pwd, context="PanSTARRS_DR2")

    query = """select o.objID, o.raStack, o.decStack, 
    m.gPSFFlux, m.gPSFFluxErr, m.gKronFlux, m.gKronFluxErr,
    m.rPSFFlux, m.rPSFFluxErr, m.rKronFlux, m.rKronFluxErr,
    m.iPSFFlux, m.iPSFFluxErr, m.iKronFlux, m.iKronFluxErr,
    m.zPSFFlux, m.zPSFFluxErr, m.zKronFlux, m.zKronFluxErr,
    m.yPSFFlux, m.yPSFFluxErr, m.yKronFlux, m.yKronFluxErr
    from fGetNearbyObjEq(%f,%f,%f) nb
    into MyDB.%s
    inner join ObjectThin o on o.objid=nb.objid
    inner join StackObjectAttributes m on o.objid=m.objid
    where m.primaryDetection=1"""%(ra, dec, radius, table_name)


    job_id = jobs.submit(query, task_name=job_name)
    while jobs.monitor(job_id)[0] != 5:
        True
    out_id = jobs.request_output(table_name, 'CSV')
    while jobs.monitor(out_id)[0] != 5:
        True
    with open(file_name, 'wb') as f:
        results = jobs.get_output(out_id, f)
    jobs.drop_table(table_name)

    table = pd.read_csv(file_name)
    table.index.name = 'index'
    table.rename(columns={'raStack':'ra', 'decStack' : 'dec'}, inplace=True)
    table.to_csv(file_name)
