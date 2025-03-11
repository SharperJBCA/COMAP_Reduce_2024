# DataCheckRemote.py 
# 
# Description: 
#  Query presto database and see how many files there are of a certain type. 

import logging 

import subprocess
from datetime import datetime
import numpy as np
import os
import stat
import pwd
import grp
import glob
from tqdm import tqdm 
import sys 
import multiprocessing
from modules.SQLModule.SQLModule import db, COMAPData 
import time 
from modules.DataAcquisition.DataAssess import DataAssess


ARG_MAX = 20971

class DataCheckRemote:
    def __init__(self, presto_info={}, sql_file_command='', local_info={}, n_rsyncs=1, dry_run=False,min_obsid_download=10000, **kwargs):
        logging.info("Initializing DataCheckRemote")

        self.sql_file_command = sql_file_command
        self.presto_info = presto_info
        self.local_info = local_info
        self.n_rsyncs = n_rsyncs
        self.dry_run = dry_run 
        self.min_obsid_download = min_obsid_download
        for k,v in kwargs.items():
            logging.info(f'{k}: {v}')
            setattr(self, k, v) 
            
    def run(self):
        logging.info("Running DataCheckRemote")

        # Step 1) Query Caltech sql database to get list of files and obsids
        logging.info('Querying SQL database at OVRO/presto')
        file_info = self.query_sql_presto()

        # Step 2) Check which obsids we already have
        logging.info('Checking which obsids we already have')
        existing_obsids = db.query_all_obsids() 
        
        filelist = np.array([info['filename'] for obsid, info in file_info.items() \
                             if not 'co' in info['target'].lower() and obsid > self.min_obsid_download])
        obsids   = np.array([obsid            for obsid, info in file_info.items() \
                             if not 'co' in info['target'].lower() and obsid > self.min_obsid_download])
        targets  = np.array([info['target']   for obsid, info in file_info.items() \
                                if not 'co' in info['target'].lower() and obsid > self.min_obsid_download])
        comments = np.array([info['comment']  for obsid, info in file_info.items() \
                                if not 'co' in info['target'].lower() and obsid > self.min_obsid_download])
        
        # Step 3) Save these arrays to disk. 
        output_filename = f'file_query_info_data/{datetime.now().strftime("%Y%m%d_%H%M%S")}_file_query_info.npz'
        np.savez(output_filename, filelist=filelist, obsids=obsids, targets=targets, existing_obsids=existing_obsids, comments=comments)

    def __str__(self):
        return "DataCheckRemote"
    
    def __repr__(self):
        return "DataCheckRemote"
    
    def query_sql_presto(self):
        """
        Query the sql database on presto to get the latest files
        """

        sql = r"""python /scr/comap_analysis/sharper/sql_query/sql_query_general.py  """
        command = ['ssh -v',self.presto_info['host'], r""" "{}" """.format(sql), r""" "{}" """.format(self.sql_file_command)]
        
        print('Running command on presto')
        p = subprocess.Popen(r""" """.join(command), shell=True, stdout=subprocess.PIPE)
        pipe, err = p.communicate()
        output = {}
        
        for iline,line in enumerate(tqdm(pipe.decode('ASCII').splitlines(),desc='Querying SQL database')):
            #obsid,_,band,relpath,path,level,filename,target_id, target_name = line.split()
            # Comment always starts with (name) comment, split line on open bracket first
            line_no_comment = line.split('(')[0]
            try:
                comment = line.split('(')[1] 
            except IndexError:
                comment = 'Unknown'
            obsid,_,band,relpath,path,level,filename = line_no_comment.split()[:7]


            #logging.info('obsid: {}, band: {}, relpath: {}, path: {}, level: {}, filename: {}'.format(obsid,band,relpath,path,level,filename))
            try:
                if isinstance(obsid,str):
                    output[int('{}'.format(obsid))] = {'filename':'/comap{}/{}/{}'.format(relpath[1:],path,filename),
                                                       'target':'Unknown','comment':comment}
                else:
                    output[int('{}'.format(obsid.decode('ASCII')))] = {'filename':'/comap{}/{}/{}'.format(relpath.decode()[1:],path.decode(),filename.decode()),
                                                                       'target':'Unknown','comment':comment}
            except ValueError:
                continue

        sql = 'python ~/sharper/sql_query/sql_query.py'
        command = ['ssh -v',self.presto_info['host'], '"{}"'.format(sql)]
        p = subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE)
        pipe, err = p.communicate()

        for line in pipe.splitlines():
            if len(line.split()) != 2:
                continue
            obsid, target = line.split()
            try:
                target = target.decode()
            except (UnicodeDecodeError, AttributeError):
                pass
            if int(obsid) in output:
                output[int(obsid)]['target'] = target

        return output