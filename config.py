"""
Set Env Variables with the correct Info
"""
import os

# RDS Postgres Database Info
username = os.environ.get("OASIS_USER")
# password = os.environ.get("OASIS_PASS")
password = 'enf_oasis!!400'
endpoint = os.environ.get("OASIS_HOST")
# data_path = os.environ.get("OASIS_DATA_PATH")
# S3 Details
# Bucket where the algo needs to be saved and imported for predictions
# bucket_name = "et-oasis"