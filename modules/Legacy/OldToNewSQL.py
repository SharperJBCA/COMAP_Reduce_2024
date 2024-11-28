# OldToNewSQL.py
#
# Description: 
#
#

import logging
import sys 
from pathlib import Path
try:
    from modules.SQLModule.SQLModule import db, COMAPData, create_feed_table
except ModuleNotFoundError:
    # This is for running the script as a standalone script
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from modules.SQLModule.SQLModule import db, COMAPData, create_feed_table


class OldToNewSQL: 

    def __init__(self,**kwargs):
        pass 

    def run(self): 
        logging.info("Running OldToNewSQL")

        # Create the feed table 
        create_feed_table() 

        # Query the old SQL database 
        old_data = db.query_all_data() 

        # Insert the old data into the new database 
        for data in old_data:
            db.insert_or_update_data(data)


if __name__ == "__main__": 
    feed_table = create_feed_table(feed_number=1, band_number=0) 

    print(feed_table.__tablename__) 
