This directory contains the code for our data pipelines.

Do not put data preprocessing scripts here; this is for workflows that
should be automated.

See `docs/AIRFLOW_GUIDE.md` for more info on how to contribute to this
directory.

Directory owner: Luke (luke_zhu@brown.edu)

## Setting up an Airflow VM
This is for if you want to create a new VM.

1. Create a VM with Python 3.6 installed. Ubuntu 18.04 LTS has this.
2. ssh into the VM with username `airflow`.
3. Clone `https://github.com/elvoai/elvo.git`
4. cd to `airflow/` and run `scripts/setup_vm.sh`
5. Add the following variables to `.bashrc`
    * SLACK_TOKEN
    * DROPBOX_TOKEN
    * GOOGLE_APPLICATION_CREDENTIALS
    * DRIVE_KEYFILE
    
For the latter 2, you will need to download JSON keyfiles from GCP.
