from google.cloud import storage

gcs_client = storage.Client.from_service_account_json(
    '../credentials/client_secret.json'
)
