import pymongo


def get_db():
    client = pymongo.MongoClient(
        "mongodb://bluenoml:elvoanalysis@104.196.51.205/elvo"
    )
    datasets = client.elvo.datasets
    return datasets
