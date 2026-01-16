from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from chatbot_with_connectedapp import ChatBot
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = ChatBot()
@app.get("/api/v1/chatbot")
async def start_conversation(prompt: str, isConnectedAppProvided: bool):
    try:
        chatbot.init_agent(isConnectedAppProvided)
        response = chatbot.qna_chatbot(prompt)
        return Response(status_code=200, content= response, media_type="text/markdown")
    except Exception as e:
        return Response(statuscode = 500, content=str(e))

if __name__ == "__main__":    
    uvicorn.run(app, host = "0.0.0.0", port = 8000)