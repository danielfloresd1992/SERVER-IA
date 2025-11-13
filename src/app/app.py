from fastapi import FastAPI


app = FastAPI()


@app.get('/', )
def init_sever():
    return { "status": "active" }

