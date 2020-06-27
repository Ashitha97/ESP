"""
Set Env Variables with the correct Info
"""
import os

# RDS Postgres Database Info
username = os.environ.get("OASIS_USER")
password = os.environ.get("OASIS_PASS")
endpoint = os.environ.get("OASIS_HOST")

# S3 Details
# Bucket where the algo needs to be saved and imported for predictions
bucket_name = "et-bonanza"