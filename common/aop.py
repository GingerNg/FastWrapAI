import time
from fastapi import HTTPException, status
from common.domains import User, db, db_app
from common.dao import FsaDao
fsa_dao = FsaDao(db, db_app)
def token_check(func):
    def wrapper(body, request, background_tasks):
        auth_token = request.headers.get("Authorization").split(" ")[1]
        # if auth_token not in context.tokens:
        user = fsa_dao.query_by_key(token=auth_token)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")
        if user.token_limit - user.token_used < 0:
            raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "cash used up!")
        result = func(body, request, background_tasks, {"user": user})
        return result
    return wrapper


## *************************** utils ********************************

def timer(func):
    """一个简单的Python装饰器，可以用来计算一个函数的执行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

if __name__ == "__main__":
    @timer
    def test_timer():
        time.sleep(1)
        print("test timer")

    test_timer()