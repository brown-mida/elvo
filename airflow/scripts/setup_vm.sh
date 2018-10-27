# You should run this only once, when first setting up airflow
# on a new VM. This works on Ubuntu 18.04 LTS with only the repo cloned

# If the last step fails you will have to add your own slack token as
# an environment variable.

#!/usr/bin/env bash

# Install Unix packages
sudo apt-get update
sudo apt-get install python3-pip python3-venv -y

# So $AIRFLOW_HOME is permanently set in the shell
echo export AIRFLOW_HOME=$PWD >> ~/.bashrc
export AIRFLOW_HOME=$PWD

# Add necessary tokens
read -p "Enter your Dropbox token:" DROPBOX_TOKEN
echo export DROPBOX_TOKEN=$DROPBOX_TOKEN >> ~/.bashrc
read -p "Enter your Slack token:" SLACK_TOKEN
echo export AIRFLOW_HOME=$SLACK_TOKEN >> ~/.bashrc


source ~/.bashrc

# Set up a virtual environment
python3 -m venv venv
source $PWD/venv/bin/activate

# Install all requirements
export AIRFLOW_GPL_UNIDECODE=yes # So we can install airflow
pip install -r requirements.txt

# Initialize airflow
airflow initdb
