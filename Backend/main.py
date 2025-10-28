import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import bs4
import uvicorn
import whisper
import tempfile
import shutil
import pyttsx3
import io
import threading
import platform
import subprocess
import hashlib
import string
import re
import base64
from collections import OrderedDict

# --- Load environment variables ---
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables or .env file.")

# --- Initialize Whisper model ---
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded successfully")

# --- Initialize TTS Engine ---
print("Initializing TTS engine...")
tts_engine = pyttsx3.init()
# Configure TTS settings
DEFAULT_TTS_RATE = 175
DEFAULT_TTS_VOLUME = 0.9
tts_engine.setProperty('rate', DEFAULT_TTS_RATE)  # Speed of speech
tts_engine.setProperty('volume', DEFAULT_TTS_VOLUME)  # Volume (0.0 to 1.0)

# Get available voices and optionally set a different voice
voices = tts_engine.getProperty('voices')
# Uncomment to use a different voice (usually index 1 is female voice)
# tts_engine.setProperty('voice', voices[1].id)
print("✅ TTS engine initialized successfully")
tts_lock = threading.Lock()


class TTSAudioCache:
    """Simple LRU cache for recently generated TTS audio blobs."""

    def __init__(self, max_items: int = 32):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._max_items = max_items

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            audio = self._cache.get(key)
            if audio is not None:
                self._cache.move_to_end(key)
            return audio

    def set(self, key: str, audio: bytes):
        with self._lock:
            self._cache[key] = audio
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_items:
                self._cache.popitem(last=False)


tts_cache = TTSAudioCache()


def make_tts_cache_key(text: str, rate: int, volume: float) -> str:
    payload = f"{rate}:{volume}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_for_repeat(text: str) -> str:
    cleaned = text.lower().translate(str.maketrans('', '', string.punctuation))
    return " ".join(cleaned.split())


EXPANSION_KEYWORDS = [
    "explain",
    "detail",
    "details",
    "why",
    "how",
    "step",
    "steps",
    "walk me through",
    "teach",
    "deep",
    "more",
]


def should_expand_answer(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in EXPANSION_KEYWORDS)


def get_tts_audio_bytes(text: str, rate: int = DEFAULT_TTS_RATE, volume: float = DEFAULT_TTS_VOLUME) -> bytes:
    cache_key = make_tts_cache_key(text, rate, volume)
    audio_data = tts_cache.get(cache_key)
    if audio_data is not None:
        return audio_data

    audio_data = text_to_speech(text, rate, volume)
    tts_cache.set(cache_key, audio_data)
    return audio_data

# --- Initialize components ---
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# --- Load and index documents (run once) ---
def initialize_vector_store():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    print("✅ Vector store initialized with documents")

# Uncomment to initialize (run once):
# initialize_vector_store()

# --- Prompts ---
qa_prompt = ChatPromptTemplate.from_template(
    """You are an expressive senior teacher guiding a curious learner. Stay strictly within the provided context.
    
    <context>
    {context}
    </context>
    
    Question: {question}
    Repeat count in this session: {repeat_count}
    
    Needs expansion: {needs_expansion}
    
    Teaching directives:
    - Sound like a real teacher without stage directions.
    - Start with one short guiding sentence.
    - If needs_expansion is "no", keep the reply to 2 concise bullet points (leading hyphens) that directly answer the question.
    - If needs_expansion is "yes", expand to 4-5 bullet points with clear steps, including one analogy or example when helpful.
    - Always end with a single-sentence comprehension check.
    - If repeat_count is greater than 0, show patient firmness: acknowledge the repetition, highlight what's new, and encourage progress.
    
    Respond in organized plain text using that outline."""
)

chat_prompt = ChatPromptTemplate.from_template(
    """You are an expressive senior teacher guiding a curious learner. Combine the context and conversation history to answer.
    
    <context>
    {context}
    </context>
    
    <conversation_history>
    {history}
    </conversation_history>
    
    Current question: {question}
    Repeat count for this question: {repeat_count}
    
    Needs expansion: {needs_expansion}
    
    Teaching directives:
    - Speak like a teacher without stage directions.
    - Begin with one short orienting sentence.
    - If needs_expansion is "no", give 2 concise bullet points (leading hyphens) that directly answer the question.
    - If needs_expansion is "yes", provide 4-5 structured bullet points with steps, weaving in an example or analogy when helpful.
    - Finish with a one-sentence comprehension check.
    - If repeat_count is greater than 0, respond with gentle firmness and highlight what the learner can focus on next.
    
    Deliver the answer in organized plain text following that outline."""
)

emotion_prompt = ChatPromptTemplate.from_template(
    """Analyze the emotional tone of this teacher's response and answer with ONLY ONE word from this list:
    happy, sad, neutral, excited, confused, frustrated, curious, satisfied, encouraging, thoughtful, firm
    
    Text: {text}
    
    Emotion:"""
)

# --- In-memory conversation storage ---
conversations = {}

# --- Pydantic models ---
class QueryRequest(BaseModel):
    question: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ResponseModel(BaseModel):
    answer: str
    emotion: Optional[str] = "neutral"
    context_used: Optional[int] = 0
    audio_base64: Optional[str] = None

class ChatResponseModel(ResponseModel):
    session_id: str
    history_length: int

class TranscriptionResponse(BaseModel):
    transcription: str
    language: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    rate: Optional[int] = DEFAULT_TTS_RATE
    volume: Optional[float] = DEFAULT_TTS_VOLUME

# --- Helper functions ---
def retrieve_context(question: str, k: int = 3) -> List[Document]:
    """Retrieve relevant documents from vector store"""
    return vector_store.similarity_search(question, k=k)

def clean_response_text(text: str) -> str:
    """Remove stage directions or markdown artifacts from the model's response."""
    stage_direction_pattern = re.compile(r"^\s*[\[(][^)\]]{0,80}[\])]\s*[:,-]*\s*", re.IGNORECASE)

    cleaned_lines = []
    for line in text.splitlines():
        stripped = stage_direction_pattern.sub("", line).strip()
        if stripped:
            cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"[ 	]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_emotion(text: str) -> str:
    """Detect emotion from text using LLM"""
    try:
        messages = emotion_prompt.invoke({"text": text})
        response = llm.invoke(messages)
        emotion = response.content.strip().lower()
        
        # Validate emotion
        valid_emotions = [
            "happy",
            "sad",
            "neutral",
            "excited",
            "confused",
            "frustrated",
            "curious",
            "satisfied",
            "encouraging",
            "thoughtful",
            "firm",
        ]
        return emotion if emotion in valid_emotions else "neutral"
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "neutral"

def generate_answer(
    question: str,
    context: List[Document],
    history: str = "",
    repeat_count: int = 0,
    needs_expansion: bool = False
) -> str:
    """Generate answer using RAG"""
    docs_content = "\n\n".join(doc.page_content for doc in context)
    expansion_flag = "yes" if needs_expansion else "no"
    
    if history:
        messages = chat_prompt.invoke({
            "question": question,
            "context": docs_content,
            "history": history,
            "repeat_count": repeat_count,
            "needs_expansion": expansion_flag
        })
    else:
        messages = qa_prompt.invoke({
            "question": question,
            "context": docs_content,
            "repeat_count": repeat_count,
            "needs_expansion": expansion_flag
        })
    
    response = llm.invoke(messages)
    return clean_response_text(response.content)

def transcribe_audio(audio_file_path: str) -> dict:
    """Transcribe audio using Whisper"""
    try:
        result = whisper_model.transcribe(audio_file_path)
        return {
            "transcription": result["text"].strip(),
            "language": result.get("language", "unknown")
        }
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def text_to_speech(text: str, rate: int = DEFAULT_TTS_RATE, volume: float = DEFAULT_TTS_VOLUME) -> bytes:
    """Convert text to speech and return audio bytes"""
    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name

        system = platform.system()

        if system == "Darwin":
            # Use macOS system TTS which reliably writes to disk
            subprocess.run(
                ["say", "-o", temp_path, "--file-format=WAVE", "--data-format=LEI16@22050", text],
                check=True
            )
        else:
            # Use global pyttsx3 engine guarded by lock for thread safety
            with tts_lock:
                tts_engine.setProperty('rate', rate)
                tts_engine.setProperty('volume', volume)
                tts_engine.save_to_file(text, temp_path)
                tts_engine.runAndWait()

        # Read the audio file
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()

        if not audio_data:
            raise ValueError("Generated audio is empty")

        # Clean up
        os.remove(temp_path)
        
        return audio_data
    except subprocess.CalledProcessError as e:
        raise Exception(f"TTS conversion failed (system command error): {str(e)}")
    except Exception as e:
        raise Exception(f"TTS conversion failed: {str(e)}")

# --- FastAPI app ---
app = FastAPI(title="Conversational RAG API with STT & TTS", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper.
    Accepts: wav, mp3, m4a, webm, ogg, flac
    """
    try:
        # Validate file type
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac']
        file_extension = os.path.splitext(audio.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(audio.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe audio
            result = transcribe_audio(temp_file_path)
            
            return TranscriptionResponse(
                transcription=result["transcription"],
                language=result["language"]
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/speak")
async def speak_endpoint(request: TTSRequest):
    """
    Convert text to speech and return audio file.
    Returns WAV audio file.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        rate = request.rate if request.rate is not None else DEFAULT_TTS_RATE
        volume = request.volume if request.volume is not None else DEFAULT_TTS_VOLUME
        audio_data = get_tts_audio_bytes(request.text, rate, volume)
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"'
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.post("/query", response_model=ResponseModel)
def query_endpoint(request: QueryRequest):
    """
    Single-turn query endpoint.
    Returns answer with emotion and context information.
    """
    try:
        # Retrieve context
        context = retrieve_context(request.question)

        # Decide level of detail
        needs_expansion = should_expand_answer(request.question)
        
        # Generate answer
        answer = generate_answer(
            request.question,
            context,
            repeat_count=0,
            needs_expansion=needs_expansion
        )

        # Detect emotion
        emotion = detect_emotion(answer)

        # Generate audio once and reuse it everywhere
        audio_b64 = None
        try:
            audio_bytes = get_tts_audio_bytes(answer)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as tts_error:
            print(f"TTS generation error (query): {tts_error}")
        
        return ResponseModel(
            answer=answer,
            emotion=emotion,
            context_used=len(context),
            audio_base64=audio_b64
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/chat", response_model=ChatResponseModel)
def chat_endpoint(request: ChatRequest):
    """
    Multi-turn conversation endpoint.
    Maintains conversation history per session_id.
    """
    try:
        # Initialize or retrieve conversation history
        if request.session_id not in conversations:
            conversations[request.session_id] = []
        
        history = conversations[request.session_id]

        normalized_question = normalize_for_repeat(request.question)
        repeat_count = sum(
            1
            for msg in history
            if msg["role"] == "user" and normalize_for_repeat(msg["content"]) == normalized_question
        )
        
        # Retrieve context
        context = retrieve_context(request.question)

        # Format history for prompt
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in history[-6:]  # Last 3 exchanges
        ])

        needs_expansion = should_expand_answer(request.question)

        # Generate answer
        answer = generate_answer(
            request.question,
            context,
            history_text,
            repeat_count,
            needs_expansion
        )

        # Detect emotion
        emotion = detect_emotion(answer)

        # Update conversation history
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": answer})

        # Keep history capped to avoid unbounded growth
        if len(history) > 40:
            del history[:-40]

        audio_b64 = None
        try:
            audio_bytes = get_tts_audio_bytes(answer)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as tts_error:
            print(f"TTS generation error (chat): {tts_error}")
        
        return ChatResponseModel(
            answer=answer,
            emotion=emotion,
            context_used=len(context),
            session_id=request.session_id,
            history_length=len(history),
            audio_base64=audio_b64
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/chat/{session_id}")
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "history": conversations[session_id],
        "message_count": len(conversations[session_id])
    }

@app.delete("/chat/{session_id}")
def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversations:
        del conversations[session_id]
        return {"message": f"Conversation history cleared for session {session_id}"}
    
    raise HTTPException(status_code=404, detail="Session not found")

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
