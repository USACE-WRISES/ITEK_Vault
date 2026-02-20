from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from local_llm import LocalLLM
import uvicorn
import os

app = FastAPI()

# ────────────────────────────────────────────────────────────────
# Use RELATIVE path — assumes itek_vectorstore folder is next to this script
# ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # folder containing query_server.py
VECTOR_STORE_FOLDER = os.path.join(BASE_DIR, "itek_vectorstore")

# Quick existence check (helpful for debugging)
if not os.path.exists(VECTOR_STORE_FOLDER):
    print(f"ERROR: Vector store folder not found at {VECTOR_STORE_FOLDER}")
    print("Make sure the 'itek_vectorstore' folder is in the same directory as this script.")
    exit(1)

print(f"Using vector store at: {VECTOR_STORE_FOLDER}")

llm = LocalLLM(
    vector_store_dir=VECTOR_STORE_FOLDER,
    query_only_mode=True,
    verbose=True
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def run_query(request: QueryRequest):
    try:
        answer, references = llm.query(request.query)
        return {"answer": answer, "references": references}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import sys
    import socket

    def find_free_port(start_port=8001):
        port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    port += 1
                    if port > start_port + 100:  # safety limit
                        raise RuntimeError("No free ports found in range 8000–8100")

    # Try preferred ports in order; fall back to auto-finding if all taken
    preferred_ports = [8001, 8000, 8080, 9000]
    selected_port = None

    for p in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                selected_port = p
                break
            except OSError:
                continue

    if selected_port is None:
        print("All preferred ports busy → searching for free port...")
        selected_port = find_free_port(8001)

    print(f"Starting server on http://127.0.0.1:{selected_port}")
    uvicorn.run(app, host="127.0.0.1", port=selected_port, log_level="info")