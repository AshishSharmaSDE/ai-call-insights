# app/main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import realtime

app = FastAPI(title="AI Call Insights")


# 👇 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only, later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(realtime.router, tags=["Realtime"])  # this defines /ws/transcribe

@app.get("/")
async def root():
    return {"message": "AI Call Insights API (POC) running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
