To set up the stuff on the department run the following commands in the
terminal.

```
git clone https://github.com/elvoai/elvo-analysis.git
cd elvo-analysis
virtualenv --python=python3.6 venv
source venv/bin/activate
pip install Keras tensorflow
```

To set up google cloud follow:
https://cloud.google.com/storage/docs/gsutil_install#linux

To access our VM run the following AFTER sshing in

```
kinit
ssh thingumy
```

Use this VM for storing and processing data.

To run the code on the GPUs, do the following
additional steps

- install tensorflow-gpu==1.4.1
- following the instructions here: https://cs.brown.edu/about/system/connecting/openvpn/osx/
- activate the VPN and ssh to `<YOUR_USERNAME>@gpu1708.cs.brown.edu