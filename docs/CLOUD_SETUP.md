This document contains instructions for using the cloud VMs.

### Commands for DevOps:
Create an instance with:
- 4 CPUs and 26 GB memory
- a GPU (may need to increase the quota)
- Ubuntu 18.04 LTS w/ 50 GB standard storage

The run the following commands
```
sudo apt-get update
sudo apt-get install gcc python-dev python-pip python3-venv
```

### First time user commands:
```
git clone https://github.com/elvoai/elvo-analysis.git # Clone the repo from GitHub
cd elvo-analysis # Change directories
python3 -m venv venv # Create a virtual environment
pip install --upgrade pip # Update pip to version 10
source venv/bin/activate # Activate the virtual environment
pip install -r requirements.txt # Install packages
```

### Every time:
On your own computer:
```
gcloud compute ssh --ssh-flag="-L 8888:localhost:8888"  <YOUR_INSTANCE_NAME>
```

On the cloud instance:
```
cd elvo-analysis
source venv/bin/activate
jupyter notebook
```

Click on the url (starts with http://localhost:8888/...)
to open the notebook.

Anything you save in the notebook will be saved on the instance.

Make sure to keep in-sync with the project using `git` fairly frequently.

### Things you should eventually know:

- How to use basic git commands (like `git add/commit/pull`)
- How to manage git branches (with `git checkout/git branch`)
- How to manage python dependencies in your virtual environemnt (with `pip`)
