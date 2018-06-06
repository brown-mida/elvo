To set up the stuff on the department run the following commands in the
terminal.

```
git clone https://github.com/elvoai/elvo-analysis.git
cd elvo-analysis
virtualenv --python=python3.6 venv
source venv/bin/activate
pip install Keras tensorflow
```

To run the code on the GPUs, do the following
additional steps

- install tensorflow-gpu==1.4.1
- following the instructions here: https://cs.brown.edu/about/system/connecting/openvpn/osx/
- activate the VPN and ssh to `<YOUR_USERNAME>@gpu1708.cs.brown.edu