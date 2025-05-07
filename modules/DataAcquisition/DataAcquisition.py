# DataAcquisition.py 
# 
# Description: 
#  The idea here is that we want to:
#   1) Query the SQL database at OVRO for the latest files 
#   2) Download these files to the local machine 
#   3) Update a local SQL database with the filepaths and some basic metadata 

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

class DataAcquisition:
    def __init__(self, presto_info={}, sql_file_command='', local_info={}, n_rsyncs=1, dry_run=False,min_obsid_download=10000, **kwargs):
        logging.info("Initializing DataAcquisition")

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
        logging.info("Running DataAcquisition")

        # Step 1) Query Caltech sql database to get list of files and obsids
        logging.info('Querying SQL database at OVRO/presto')
        files = self.query_sql_presto()

        # Step 2) Check which obsids we already have
        logging.info('Checking which obsids we already have')
        existing_obsids = db.query_all_obsids() 
        
        logging.info('Running rsyncs')
        local_paths = self.run_rsyncs(files,existing_obsids,dry_run=self.dry_run,num_rsyncs=self.n_rsyncs)

        logging.info('Updating permissions')
        self.update_permissions(local_paths)

    def __str__(self):
        return "DataAcquisition"
    
    def __repr__(self):
        return "DataAcquisition"
    
    def query_sql_presto(self):
        """
        Query the sql database on presto to get the latest files
        """

        sql = r"""python /scr/comap_analysis/sharper/sql_query/sql_query_general.py  """
        command = ['ssh -v',self.presto_info['host'], r""" "{}" """.format(sql)]
        
        print('Running command on presto')
        p = subprocess.Popen(r""" """.join(command), shell=True, stdout=subprocess.PIPE)
        pipe, err = p.communicate()
        output = {}
        for iline,line in enumerate(tqdm(pipe.decode('ASCII').splitlines(),desc='Querying SQL database')):
            #obsid,_,band,relpath,path,level,filename,target_id, target_name = line.split()
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
                                                                       'target':'Unknown', 'comment':comment}
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

    def run_rsyncs(self, file_info, existing_obsids, num_rsyncs=5, dry_run=True):
        """
        Calls rsync to remote server, can return filelist
        """
        if dry_run:
            dry_run_str = '--dry-run'
        else:
            dry_run_str = ''

        filelist = np.array([info['filename'] for obsid, info in file_info.items() if not 'co' in info['target'].lower() and obsid not in existing_obsids and obsid > self.min_obsid_download])
        obsids   = np.array([obsid            for obsid, info in file_info.items() if not 'co' in info['target'].lower() and obsid not in existing_obsids and obsid > self.min_obsid_download])
        local_paths = np.array([os.path.join(self.local_info['data_directory'],os.path.basename(info['filename'])) for obsid, info in file_info.items() if not 'co' in info['target'].lower() and obsid not in existing_obsids and obsid > self.min_obsid_download])
        
        print(f'{filelist.size} files to download')
        logging.info(f'{filelist.size} files to download')

        rsync_jobids = np.sort(np.mod(np.arange(filelist.size),num_rsyncs))
        chunks = np.array_split(np.arange(filelist.size), num_rsyncs)
        base_command = 'rsync -rLptgoDvhz -e ssh {} {} --progress {} --bwlimit=5000'

        def download_with_retry(remote_path, local_path, max_retries=2):
            """Download a single file with retries"""
            for i in range(max_retries):
                try:
                    command = base_command.format(remote_path, local_path, dry_run_str)
                    process = subprocess.run(command, shell=True, check=True)
                    return True
                except Exception as e:
                    logging.info(f'Attempt {i+1} failed')
                    logging.error(f"Error downloading file: {str(e)}")
                    if i < max_retries - 1:
                        sleep_time = (i+1) * 10 
                        logging.info(f"Sleeping for {sleep_time} seconds before retrying")
                        time.sleep(sleep_time)
                    else:
                        logging.error(f"Failed to download file after {max_retries} attempts")
                        return False
                    
        def download_and_update(file_indices):
            """Download files and update database for a group of files"""
            successful_downloads = [] 
            failed_downloads     = []
            for idx in file_indices:
                # Download single file
                remote_path = f"{self.presto_info['host']}:{filelist[idx]}"

                if download_with_retry(remote_path, self.local_info['data_directory']):
                    successful_downloads.append(idx)

                    try: 
                        single_local_path = [local_paths[idx]]
                        single_obsid = [obsids[idx]]
                        file_info = DataAssess.get_file_paths_metadata(single_local_path)
                        DataAssess.update_sql_database(single_obsid, file_info)
                        logging.info(f"Successfully downloaded and updated database for obsid {single_obsid}")
                    except Exception as e:
                        logging.error(f"Error updating database for obsid {single_obsid}: {str(e)}")
                        failed_downloads.append(idx)
                        successful_downloads.remove(idx)
                else:
                    logging.error(f"Failed to download file for obsid {obsids[idx]}")
                    failed_downloads.append(idx)

            
            return successful_downloads, failed_downloads

        all_successful_downloads = []
        all_failed_downloads = []
        result_queue = multiprocessing.Queue()

        processes = []
        for chunk in chunks:
            p = multiprocessing.Process(target=lambda q, chunk: q.put(download_and_update(chunk)), args=(result_queue, chunk))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        while not result_queue.empty():
            successful_downloads, failed_downloads = result_queue.get()
            all_successful_downloads.extend(successful_downloads)
            all_failed_downloads.extend(failed_downloads)

        logging.info(f"Successfully downloaded and updated database for {len(all_successful_downloads)} files")
        logging.info(f"Failed to download and update database for {len(all_failed_downloads)} files")

        return local_paths

    def update_permissions(self,paths):
        gid = grp.getgrnam('comap').gr_gid
        uid = os.getuid()

        for filename in paths:
            try:
                os.chmod(f'{filename}',(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH) )
                os.chown(f'{filename}',uid,gid)
            except OSError:
                continue