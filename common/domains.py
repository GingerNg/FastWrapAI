from dataclasses import dataclass, asdict
from typing import List, Dict
import datetime
from utils.flask_sql_utils import init_db_app
db_url = f"sqlite:///XXXXXX.sqlite3"
db, db_app = init_db_app(db_url=db_url)

@dataclass
class Dialog(db.Model):
    payload: List
    message: Dict
    user_id: str
    model: str
    token_used: int

    id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    create_time = db.Column(db.DateTime,
                           default=datetime.datetime.now, nullable=False)
    update_time = db.Column(db.DateTime,
                           default=datetime.datetime.now,
                           onupdate=datetime.datetime.now, nullable=False)
    payload = db.Column(db.JSON)
    message = db.Column(db.JSON)
    user_id = db.Column(db.Integer)
    model = db.Column(db.String(128))
    token_used = db.Column(db.Integer)

    def __init__(self, payload, message, user_id, model, token_used=-1):
        self.payload = payload
        self.message = message
        self.user_id = user_id
        self.model = model
        self.token_used = token_used


@dataclass
class User(db.Model):
    name: str
    token: str
    token_limit: int
    token_used: int

    id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    create_time = db.Column(db.DateTime,
                           default=datetime.datetime.now, nullable=False)
    update_time = db.Column(db.DateTime,
                           default=datetime.datetime.now,
                           onupdate=datetime.datetime.now, nullable=False)
    name = db.Column(db.String(32))
    token = db.Column(db.String(128))
    token_limit = db.Column(db.Integer)
    token_used = db.Column(db.Integer)

    def __init__(self, name, token, token_limit, token_used=0):
        self.name = name
        self.token = token
        self.token_limit = token_limit
        self.token_used = token_used

UserTokens = {}
UserTokens["XXXX"] = User(name="test", token="XXXX", token_limit=10000)
UserTokens["XXXX"] = User(name="test", token="XXXX", token_limit=10000)

if __name__ == "__main__":
    user = User(user="test", token="test", token_limit=100)
    print(user)
    print(asdict(user))