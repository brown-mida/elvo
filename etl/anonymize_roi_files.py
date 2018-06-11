"""rename the numeric files in the Dropbox Folder "ROI_cropped"
to the corresponding anonymized alphanumber ID found in sheet 3 of
Google Drive folder

Note: since this is a one-time change, I (Mary) just hard-coded the path
according to my laptop's directory structure.
"""

import csv
import os
# open and store the csv file
IDs = {}
with open('/Users/mdong/Desktop/ELVO_Key - Sheet3.csv', 'rb') as csvfile:
    timeReader = csv.reader(csvfile, delimiter=',')
    # build dictionary with associated IDs
    for row in timeReader:
        IDs[row[0]] = row[1]

# move files
path = '/Users/mdong/Dropbox (Brown)/ROI_ELVOs/ROI_cropped'
tmpPath = '/Users/mdong/Dropbox (Brown)/ROI_ELVOs/ROI_cropped_anon'
for oldname in os.listdir(path):
    # ignore files in path which aren't in the csv file
    if oldname in IDs:
       os.rename(os.path.join(path, oldname),
                 os.path.join(tmpPath, IDs[oldname]))
    elif oldname.replace("_1", "") in IDs:
        oldname_wo_number = oldname.replace("_1", "")
        os.rename(os.path.join(path, oldname),
                  os.path.join(tmpPath, (IDs[oldname_wo_number] + "_1")))
    elif oldname.replace("_2", "") in IDs:
        oldname_wo_number = oldname.replace("_2", "")
        os.rename(os.path.join(path, oldname),
                  os.path.join(tmpPath, (IDs[oldname_wo_number] + "_2")))
    elif oldname.replace("_3", "") in IDs:
        oldname_wo_number = oldname.replace("_3", "")
        os.rename(os.path.join(path, oldname),
                  os.path.join(tmpPath, (IDs[oldname_wo_number] + "_3")))
