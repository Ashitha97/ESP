"""
Set Env Variables with the correct Info
"""
import os

# RDS Postgres Database Info
username = os.environ.get("OASIS_USER")
password = os.environ.get("OASIS_PASS")
endpoint = os.environ.get("OASIS_HOST")
