FROM python:3.6
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY preprocessing .
CMD ["python", "preprocess.py"]