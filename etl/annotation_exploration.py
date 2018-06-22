"""
Programmatically downloads annotation data in
https://www.dropbox.com/home/ELVOs_anon for exploration.

To run this code type the following commands in your terminal:
    conda install -c conda-forge dropbox

Before running, set the environment variable DROPBOX_TOKEN to an
access token.
"""
import os

import dropbox
from dropbox.files import FolderMetadata

if __name__ == '__main__':

    os.environ['DROPBOX_TOKEN'] = \
        '7rhGNlTWWHAAAAAAAAAAJgYJurAsZ3H_A7GDKYgRzAqXuBv4dNO0CCyizIVFDb6A'

    # TODO: Use a config file to store env variables
    # to store secret info
    dbx = dropbox.Dropbox(os.environ['DROPBOX_TOKEN'])

    results = dbx.files_list_folder('/home/ROI_ELVOs/'
                                    'ROI_cropped_anon/AEPRN5R7W2ASOGR0')

    print(results)

    if results.has_more:
        raise RuntimeError('has_more=True is currently not supported')

    for entry in results.entries:
        if isinstance(entry, FolderMetadata):
            print('ignoring folder:', entry)
            continue
        print('downloading from dropbox:', entry.name)
        # response = dbx.files_download_to_file(entry.name, entry.id)
        # print('uploading to GCS:', 'ELVOs_anon/' + entry.name)
