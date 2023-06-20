from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from utils.env_utils import app
from dataclasses import dataclass, asdict
from common.domains import User, db, db_app
from common.dao import FsaDao
fsa_dao = FsaDao(db, db_app)

class UserBody(BaseModel):
    name: str
    token_limit : int = 10000

admin_token = "XXXXXX"

@app.post("/user/register")
async def register(body: UserBody, request: Request, background_tasks: BackgroundTasks):
    print(request.headers)
    auth_token = request.headers.get("authorization").split(" ")[1]
    if auth_token != admin_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    user = User(name=body.name, token=body.name+"-hacker", token_limit=body.token_limit)
    user_obj = asdict(user)
    fsa_dao.save_obj(user)
    return JSONResponse(status_code=200, content={"message": "success", "data": user_obj})



@app.post("/user/query_amount")
async def register(body: UserBody, request: Request, background_tasks: BackgroundTasks):
    user = fsa_dao.query_by_key(body.name)
    return JSONResponse(status_code=200, content={"message": "success", "data": {"amount": user.token_limit, "used": user.token_used}})