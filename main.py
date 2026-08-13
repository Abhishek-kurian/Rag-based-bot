from __future__ import annotations
import os,secrets,time
from collections import OrderedDict
from dataclasses import asdict
from fastapi import FastAPI,File,HTTPException,UploadFile,Cookie
from fastapi.responses import FileResponse,JSONResponse
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from rag import SessionVectorStore,parse_document,LLMService,build_context,build_chatml_prompt

load_dotenv()
MAX_UPLOAD_MB=int(os.getenv("MAX_UPLOAD_MB","15"));MAX_UPLOAD_BYTES=MAX_UPLOAD_MB*1024*1024
ALLOWED={".pdf",".txt",".md"};MAX_SESSIONS=100
app=FastAPI(title="Qwen RAG Chatbot",version="1.0.0")
class SessionState:
    def __init__(self):
        self.created_at=time.time();self.last_seen=time.time();self.store=SessionVectorStore()
        self.documents=OrderedDict();self.chat_history=[];self.user_hf_token=None
SESSIONS=OrderedDict()
class ChatRequest(BaseModel):
    query:str=Field(min_length=1,max_length=8000);hf_token:str|None=Field(default=None,max_length=500)
class TokenRequest(BaseModel):hf_token:str|None=Field(default=None,max_length=500)

def session(sid):
    if sid and sid in SESSIONS:
        x=SESSIONS.pop(sid);x.last_seen=time.time();SESSIONS[sid]=x;return sid,x
    sid=secrets.token_urlsafe(32);x=SessionState();SESSIONS[sid]=x
    while len(SESSIONS)>MAX_SESSIONS:SESSIONS.popitem(last=False)
    return sid,x
def cookie(r,sid):r.set_cookie("rag_session",sid,httponly=True,secure=False,samesite="lax",max_age=43200)
def filename(name):
    name=os.path.basename(name or "").strip();ext=os.path.splitext(name)[1].lower()
    if not name:raise HTTPException(400,"Filename is required.")
    if ext not in ALLOWED:raise HTTPException(400,"Only PDF, TXT and MD are supported.")
    return name

@app.get("/")
async def index():return FileResponse("static/index.html")
@app.get("/health")
async def health():return {"status":"ok","llm_model":os.getenv("LLM_MODEL","Qwen/Qwen2.5-3B-Instruct"),
    "embedding_model":os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")}
@app.post("/api/session")
async def create(rag_session:str|None=Cookie(default=None)):
    sid,s=session(rag_session);r=JSONResponse({"session_id":sid,"documents":len(s.documents),"chunks":s.store.count});cookie(r,sid);return r
@app.post("/api/upload")
async def upload(files:list[UploadFile]=File(...),rag_session:str|None=Cookie(default=None)):
    sid,s=session(rag_session);results=[]
    for f in files:
        name=filename(f.filename);data=await f.read()
        if len(data)>MAX_UPLOAD_BYTES:raise HTTPException(413,f"{name} exceeds {MAX_UPLOAD_MB} MB.")
        try:
            chunks=parse_document(data,name)
            added=s.store.add_chunks(chunks)
            if added:
                s.documents[name]={"filename":name,"chunks":added,"size_bytes":len(data)}
                results.append({"filename":name,"status":"indexed","chunks":added})
            else:results.append({"filename":name,"status":"skipped","message":"No readable text or chunk limit reached."})
        except Exception as e:results.append({"filename":name,"status":"error","message":str(e)})
    r=JSONResponse({"session_id":sid,"results":results,"documents":list(s.documents.values()),"total_chunks":s.store.count});cookie(r,sid);return r
@app.get("/api/documents")
async def docs(rag_session:str|None=Cookie(default=None)):
    _,s=session(rag_session);return {"documents":list(s.documents.values()),"total_chunks":s.store.count}
@app.get("/api/chunks")
async def chunks(limit:int=100,rag_session:str|None=Cookie(default=None)):
    _,s=session(rag_session);limit=max(1,min(limit,200));return {"chunks":[asdict(x) for x in s.store.chunks[:limit]],"total":s.store.count}
@app.post("/api/token")
async def save_token(p:TokenRequest,rag_session:str|None=Cookie(default=None)):
    sid,s=session(rag_session);s.user_hf_token=p.hf_token.strip() if p.hf_token else None
    r=JSONResponse({"session_id":sid,"configured":bool(s.user_hf_token)});cookie(r,sid);return r
@app.delete("/api/token")
async def del_token(rag_session:str|None=Cookie(default=None)):
    _,s=session(rag_session);s.user_hf_token=None;return {"configured":False}
@app.post("/api/chat")
async def chat(p:ChatRequest,rag_session:str|None=Cookie(default=None)):
    sid,s=session(rag_session);q=p.query.strip()
    token=p.hf_token.strip() if p.hf_token else s.user_hf_token
    if p.hf_token:s.user_hf_token=p.hf_token.strip()
    retrieved=s.store.search(q,5);context=build_context(retrieved)
    try:answer,provider=LLMService().generate(q,context,token)
    except Exception as e:raise HTTPException(503,str(e))
    s.chat_history.extend([{"role":"user","content":q},{"role":"assistant","content":answer}]);s.chat_history=s.chat_history[-20:]
    return {"session_id":sid,"answer":answer,"provider":provider,"retrieved":[
        {"rank":i+1,"score":round(x.score,4),"document":x.chunk.document_name,"page":x.chunk.page,
         "chunk_id":x.chunk.chunk_id,"text":x.chunk.text} for i,x in enumerate(retrieved)],
        "raw_prompt":build_chatml_prompt(q,context)}
@app.delete("/api/chat")
async def clear_chat(rag_session:str|None=Cookie(default=None)):
    _,s=session(rag_session);s.chat_history.clear();return {"status":"cleared"}
@app.delete("/api/documents")
async def clear_docs(rag_session:str|None=Cookie(default=None)):
    _,s=session(rag_session);s.store.clear();s.documents.clear();return {"status":"cleared","chunks":0}
