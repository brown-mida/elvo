[![Build Status](https://travis-ci.com/elvoai/elvo-analysis.svg?branch=master)](https://travis-ci.com/elvoai/elvo-analysis)
[![codecov](https://codecov.io/gh/elvoai/elvo-analysis/branch/master/graph/badge.svg)](https://codecov.io/gh/elvoai/elvo-analysis)
[![Documentation Status](https://readthedocs.org/projects/elvo-analysis/badge/?version=latest)](https://elvo-analysis.readthedocs.io/en/latest/?badge=latest)

### Getting started
Create a virtual environment, run `pip install -e .`
and then run `pytest`. If the test cases pass you should be good to
start developing.

### Organization

#### Folder Organization

The project contains the following core folders:

- `credentials/` # The other secrets folder
- `blueno/` # Shared code for our ML platform
- `dashboard/` # Code for the ELVO App Engine dashboard
- `data/` # Contains all downloaded data. Data used in the code should be stored in the cloud.
- `docs/` # Documentation for the project.
- `etl/` # All data pipeline scripts, code which should be scheduled on Airflow
- `logs/` # For storing application logs
- `ml/` # All ML specific scripts, contains the blueno ML tookit as well
- `models/` # For storing trained ML models (HDF5, etc.)
- `notebooks/` # Contains all notebooks.
- `secrets/` # Store your secrets in this folder, so they don’t get uploaded to GitHub.

`dashboard`, `etl`, and `ml` should be seen as top-level python
projects.
This means each folder should contain top level scripts with
packages as sub-folders.
As our codebase is small, we are keeping them in a single repo but there
are plans to separate these in the future.


### Contributing
To contribute, create a pull request. Every PR should
 be reviewed by at least one other person. See
[this gist](https://gist.github.com/kashifrazzaqui/44b868a59e99c2da7b14)
for a guide on how to review a PR.

### Other

For developing on GPUs. See the `docs` folder for more info.