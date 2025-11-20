from fastapi import FastAPI

app = FastAPI(
    title="Multimodal Financial Market Intelligence Agent",
    description="Processes audio, video, stock, news, and balance sheet data for RAG.",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to the Multimodal Financial Market Intelligence Agent 🚀"}

