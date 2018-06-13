To set up the department GPU run the following commands in the
terminal. Make sure that Tunnelblick is running:

```
ssh -L 8888:localhost:8888 <YOUR USERNAME>@gpu1708.cs.brown.edu`
```

Run these commands on `gpu1708`.
```
git clone https://github.com/elvoai/elvo-analysis.git
cd elvo-analysis
virtualenv --python=python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
pip install tensorflow-gpu=1.4.1
````

To get new data, you should use `gsutil` to copy the data
to the deparment filesystem (to the path which your code
points to).

To set up `gsutil` follow:
https://cloud.google.com/storage/docs/gsutil_install#linux

To access our VM run the following AFTER sshing in

```
kinit
ssh thingumy
```

We can use this VM for storing and processing data.