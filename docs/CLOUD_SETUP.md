This document contains instructions for using the cloud VMs.

### Commands for DevOps:
Create an instance with:
- us-east-1-b
- 4 CPUs and 26 GB memory
- a GPU (may need to increase the quota)
- Ubuntu 17.10 w/ 50 GB standard storage

The run the following commands
```
sudo apt-get update
sudo apt-get install -y gcc python-dev python-pip python3.6-venv
```

Then follow the instructions here: `https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver`
to install a gpu driver

```
#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 17.04 installer works with 17.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
  sudo dpkg -i ./cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install cuda-9-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1
```

### First time user commands:
```
git clone https://github.com/elvoai/elvo-analysis.git # Clone the repo from GitHub
cd elvo-analysis # Change directories
python3 -m venv venv # Create a virtual environment
pip install --upgrade pip # Update pip to version 10
source venv/bin/activate # Activate the virtual environment
pip install -r requirements.txt # Install packages
pip install tensorflow-gpu==1.8.0 # Install tensorflow for the gpu
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
