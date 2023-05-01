import modules
from fastapi import FastAPI
from pydantic import BaseModel

# print(modules.model_create('text_file.csv'))
# print('')
# data=f"ðŸ”¥ https://fans.ly/r/Vikidream Oooo I wish you were here at my place, If you were here, I would give you a long MASSAGE and you could return the FAVOUR."
# print(modules.predict(data))

app = FastAPI()
file="text_file.csv"
@app.post("/filter")
async def questionAnswer(data:str):
    # print(type(data))
    value=modules.predict(data)
    # print(value)
    return {"Spam: ":str(value)}

