To run the code, clone the repository, download the data from the Dropbox so
the data is in the same folder

```
some_directory/
    RI Hospital ELVO DATA/...
    Example Scan.ipynb
    untitled.py
```
 
Then run the following commands while
in the root directory of the repo.

```
conda install pydicom
conda install pytorch torchvision -c pytorch
python3 main.py
```

Note: We are using Anaconda >4.4 (for Python 3.6)

The Jupyter notebooks contains some code for generating images of
the data which can be interactively run in the browser.

Type in `jupyter notebook` into the console to start the notebook.


## TODO

* Document the schema of the processed data (labels, centerX, etc.)
* Set up the validation pipeline: See
    http://cs231n.github.io/neural-networks-3/
    https://research.googleblog.com/2017/11/feature-visualization.html
* Write a few system tests for checking the preprocessing step
* Write better instructions for how to get started with the code
* Create a contribution guide
* Make sure that the data is anonymized.

Mary Dong

Andrew Kim
