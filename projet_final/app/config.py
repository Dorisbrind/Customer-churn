import os


## Database config

DEVELOPMENT = True
DEBUG = True
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", default="postgresql://postgres:19951209@127.0.0.1:5432/md4-projet2")