import os
import dropbox


def authenticate(token):
    return dropbox.Dropbox(token)


def get_cab_from_folder(dbx, dbx_folder):
    if dbx_folder[0] != '/':
        dbx_folder = '/' + dbx_folder
    response = dbx.files_list_folder(dbx_folder)

    files = []
    for file in response.entries:
        if file.name.split('.')[-1] == 'cab':
            files.append(file.name)
    while response.has_more:
        response = dbx.files_list_folder_continue(response.cursor)
        for file in response.entries:
            if file.name.split('.')[-1] == 'cab':
                files.append(file.name)
    return files


def download_from_dropbox(dbx, tmp_folder, dbx_folder, filename):
    if dbx_folder[0] != '/':
        dbx_folder = '/' + dbx_folder
    return dbx.files_download_to_file(os.path.join(tmp_folder, filename),
                                      os.path.join(dbx_folder, filename))
