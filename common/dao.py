from .domains import User
class Dao(object):
    def __init__(self) -> None:
        pass

class FsaDao(Dao):
    # falsk sqlac dao
    def __init__(self, db, app) -> None:
        self.app = app
        self.db = db
        self._create_db()

    def save_obj(self, obj):
        with self.app.app_context():
            self.db.session.add(obj)
            self.db.session.commit()

    def query_by_key(self, name=None, token=None) -> User:
        item = None
        if name:
            print(name)
            with self.app.app_context():
                item = User.query.filter_by(name=name).first()
        elif token:
            with self.app.app_context():
                item = User.query.filter_by(token=token).first()
        return item

    def _create_db(self):
        with self.app.app_context():
            self.db.create_all()