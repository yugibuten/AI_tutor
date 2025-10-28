import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Trash2,
  Mic,
  Square,
  MessageSquare,
  Sparkles,
  VolumeX,
  Volume2,
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8001';
const DEFAULT_RATE = 175;
const DEFAULT_VOLUME = 0.9;

const BASE_EXPRESSION = {
  theme: 'emotion-neutral',
  label: 'Neutral',
  eyeClass: '',
  browClass: '',
  mouthClass: '',
};

const createExpression = (overrides) => ({
  ...BASE_EXPRESSION,
  ...overrides,
});

const EMOTION_EXPRESSIONS = {
  neutral: createExpression({}),
  happy: createExpression({
    theme: 'emotion-happy',
    label: 'Happy',
    eyeClass: 'bright',
    browClass: 'relaxed',
    mouthClass: 'smile',
  }),
  excited: createExpression({
    theme: 'emotion-happy',
    label: 'Excited',
    eyeClass: 'bright',
    browClass: 'raised',
    mouthClass: 'smile',
  }),
  satisfied: createExpression({
    theme: 'emotion-happy',
    label: 'Satisfied',
    eyeClass: 'bright',
    browClass: 'relaxed',
    mouthClass: 'smile',
  }),
  sad: createExpression({
    theme: 'emotion-sad',
    label: 'Sad',
    eyeClass: 'droop',
    browClass: 'sad',
    mouthClass: 'frown',
  }),
  frustrated: createExpression({
    theme: 'emotion-frustrated',
    label: 'Frustrated',
    eyeClass: 'sharp',
    browClass: 'angry',
    mouthClass: 'flat',
  }),
  confused: createExpression({
    theme: 'emotion-curious',
    label: 'Confused',
    eyeClass: 'tilt',
    browClass: 'raised',
    mouthClass: 'flat',
  }),
  curious: createExpression({
    theme: 'emotion-curious',
    label: 'Curious',
    eyeClass: 'tilt',
    browClass: 'raised',
    mouthClass: 'flat',
  }),
  encouraging: createExpression({
    theme: 'emotion-happy',
    label: 'Encouraging',
    eyeClass: 'bright',
    browClass: 'raised',
    mouthClass: 'smile',
  }),
  thoughtful: createExpression({
    theme: 'emotion-curious',
    label: 'Thoughtful',
    eyeClass: 'tilt',
    browClass: 'relaxed',
    mouthClass: 'flat',
  }),
  firm: createExpression({
    theme: 'emotion-firm',
    label: 'Firm',
    eyeClass: 'sharp',
    browClass: 'angry',
    mouthClass: 'flat',
  }),
};

function Mascot({ emotion, isListening, isThinking, isTranscribing, isSpeaking }) {
  const expression = EMOTION_EXPRESSIONS[emotion] || EMOTION_EXPRESSIONS.neutral;

  let statusMessage = 'Ready to chat!';
  if (isListening) {
    statusMessage = '🎤 Listening...';
  } else if (isTranscribing) {
    statusMessage = '⏳ Processing your speech...';
  } else if (isThinking) {
    statusMessage = '🤔 Thinking...';
  } else if (isSpeaking) {
    statusMessage = '🔊 Speaking...';
  } else {
    statusMessage = `Feeling ${expression.label.toLowerCase()}.`;
  }

  return (
    <div className="mascot-panel-content">
      <div className="mascot-title">
        <div className="mascot-title-icon">
          <Sparkles size={24} />
        </div>
        <div>
          <h1>AI Tutor Mascot</h1>
          <p>Talk or type to learn about LLM agents.</p>
        </div>
      </div>

      <div className="mascot-face-wrapper">
        <div className={`mascot-face ${expression.theme}`}>
          <div className={`mascot-brow left ${expression.browClass}`} />
          <div className={`mascot-brow right ${expression.browClass}`} />
          <div className={`mascot-eye left ${expression.eyeClass}`} />
          <div className={`mascot-eye right ${expression.eyeClass}`} />
          <div className={`mascot-mouth ${expression.mouthClass} ${isSpeaking ? 'speaking' : ''}`} />
        </div>
      </div>

      <div className="mascot-status">{statusMessage}</div>
    </div>
  );
}

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [mode, setMode] = useState('chat');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState('neutral');
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);
  const audioUrlRef = useRef(null);

  useEffect(() => {
    setSessionId(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const stopSpeaking = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    setIsSpeaking(false);
  };

  const fetchSpeechBlobFromServer = async (text) => {
    if (!text) return null;

    const response = await fetch(`${API_BASE_URL}/speak`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        rate: DEFAULT_RATE,
        volume: DEFAULT_VOLUME,
      }),
    });

    if (!response.ok) {
      throw new Error(`TTS failed: ${response.status}`);
    }

    return await response.blob();
  };

  const playSpeechBlob = async (audioBlob) => {
    if (!audioBlob) return;

    stopSpeaking();
    const audioUrl = URL.createObjectURL(audioBlob);
    audioUrlRef.current = audioUrl;

    if (audioRef.current) {
      audioRef.current.src = audioUrl;
      audioRef.current.onended = () => {
        setIsSpeaking(false);
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current);
          audioUrlRef.current = null;
        }
      };
      setIsSpeaking(true);
      await audioRef.current.play();
    }
  };

  const playAudioFromBase64 = async (audioBase64) => {
    if (!audioBase64) return;

    try {
      const byteString = atob(audioBase64);
      const buffer = new ArrayBuffer(byteString.length);
      const view = new Uint8Array(buffer);
      for (let i = 0; i < byteString.length; i += 1) {
        view[i] = byteString.charCodeAt(i);
      }

      const audioBlob = new Blob([buffer], { type: 'audio/wav' });
      await playSpeechBlob(audioBlob);
    } catch (error) {
      console.error('Failed to play base64 audio:', error);
      setIsSpeaking(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(audioBlob);

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Unable to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const transcribeAudio = async (audioBlob) => {
    setIsTranscribing(true);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');

      const response = await fetch(`${API_BASE_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.status}`);
      }

      const data = await response.json();

      if (data.transcription) {
        await handleSendMessage(data.transcription);
      } else {
        alert('No speech detected. Please try again.');
      }
    } catch (error) {
      console.error('Transcription error:', error);
      alert(`Transcription failed: ${error.message}. Make sure the backend is running.`);
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleSendMessage = async (messageText = null) => {
    const textToSend = (messageText ?? input).trim();
    if (!textToSend) return;

    const userMessage = {
      role: 'user',
      content: textToSend,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const endpoint = mode === 'chat' ? '/chat' : '/query';
      const payload =
        mode === 'chat'
          ? { session_id: sessionId, question: textToSend }
          : { question: textToSend };

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        emotion: data.emotion,
        context_used: data.context_used,
        timestamp: new Date().toISOString(),
        audioBase64: data.audio_base64 || null,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setCurrentEmotion(data.emotion || 'neutral');
      if (data.audio_base64) {
        await playAudioFromBase64(data.audio_base64);
      } else {
        try {
          const fallbackBlob = await fetchSpeechBlobFromServer(data.answer);
          await playSpeechBlob(fallbackBlob);
        } catch (ttsError) {
          console.error('TTS fetch error:', ttsError);
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setCurrentEmotion('frustrated');
      const errorMessage = {
        role: 'assistant',
        content: `❌ Error: ${error.message}. Make sure the backend is running on port 8001.`,
        emotion: 'frustrated',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = () => {
    handleSendMessage();
  };

  const handleReplayAudio = async (message) => {
    if (!message || message.role !== 'assistant') return;
    try {
      if (message.audioBase64) {
        await playAudioFromBase64(message.audioBase64);
      } else {
        const audioBlob = await fetchSpeechBlobFromServer(message.content);
        await playSpeechBlob(audioBlob);
      }
    } catch (error) {
      console.error('Replay audio error:', error);
      setIsSpeaking(false);
    }
  };

  const handleClearConversation = async () => {
    try {
      if (mode === 'chat') {
        await fetch(`${API_BASE_URL}/chat/${sessionId}`, {
          method: 'DELETE',
        });
      }
      stopSpeaking();
      setMessages([]);
      setCurrentEmotion('neutral');
      setSessionId(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    } catch (error) {
      console.error('Error clearing conversation:', error);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const lastAssistantMessage = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant');

  return (
    <div className="app-shell">
      <audio ref={audioRef} style={{ display: 'none' }} />

      <div className="mascot-panel">
        <Mascot
          emotion={currentEmotion}
          isListening={isRecording}
          isThinking={loading}
          isTranscribing={isTranscribing}
          isSpeaking={isSpeaking}
        />

        <div className="voice-controls">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={loading || isTranscribing}
            className={`voice-button primary ${isRecording ? 'recording' : ''}`}
          >
            {isRecording ? <Square size={20} /> : <Mic size={20} />}
            <span>{isRecording ? 'Stop listening' : 'Talk to the mascot'}</span>
          </button>

          <button
            onClick={stopSpeaking}
            disabled={!isSpeaking}
            className="voice-button secondary"
          >
            <VolumeX size={18} />
            <span>Stop audio</span>
          </button>
        </div>

        <div className="mascot-dialog">
          <h3>Latest reply</h3>
          <p>
            {lastAssistantMessage
              ? lastAssistantMessage.content
              : 'Say hello or type a question to start the conversation.'}
          </p>
        </div>
      </div>

      <div className="chat-panel">
        <div className="chat-header">
          <div className="chat-title">
            <div className="chat-icon">
              <MessageSquare size={20} />
            </div>
            <div>
              <h2>Conversation</h2>
              <p>{mode === 'chat' ? 'Multi-turn session' : 'Single-turn answer'}</p>
            </div>
          </div>

          <div className="chat-header-actions">
            <div className="mode-toggle">
              <button
                className={mode === 'chat' ? 'active' : ''}
                onClick={() => setMode('chat')}
              >
                Multi-turn
              </button>
              <button
                className={mode === 'query' ? 'active' : ''}
                onClick={() => setMode('query')}
              >
                Single
              </button>
            </div>
            <button
              onClick={handleClearConversation}
              className="clear-button"
              title="Clear conversation"
            >
              <Trash2 size={18} />
              Clear
            </button>
          </div>
        </div>

        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <Sparkles size={48} />
              <h3>Welcome!</h3>
              <p>
                Talk to the mascot or type a question about LLM agents and task decomposition.
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={`message-row ${message.role === 'user' ? 'user' : 'assistant'}`}
              >
                <div
                className={`message-bubble ${
                    message.role === 'user' ? 'user-bubble' : 'assistant-bubble'
                  }`}
                >
                  <p>{message.content}</p>
                  {message.role === 'assistant' && (
                    <div className="message-footer">
                      {message.emotion && (
                        <span className={`emotion-badge emotion-${message.emotion}`}>
                          {message.emotion}
                        </span>
                      )}
                      {message.context_used > 0 && (
                        <span className="context-info">{message.context_used} sources</span>
                      )}
                      <button
                        onClick={() => handleReplayAudio(message)}
                        className="speak-inline"
                        disabled={isSpeaking}
                      >
                        <Volume2 size={14} />
                        Hear it again
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="message-row assistant">
              <div className="message-bubble assistant-bubble loading">
                <div className="loading-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isTranscribing ? 'Transcribing...' : 'Type your question...'}
            disabled={loading || isTranscribing || isSpeaking}
            rows={1}
          />

          <div className="chat-input-actions">
            <button
              onClick={handleSend}
              disabled={loading || !input.trim() || isTranscribing || isSpeaking}
              className="send-button"
            >
              <Send size={18} />
              Send
            </button>
          </div>

          <div className="chat-input-status">
            {isRecording && '🔴 Recording... click the mic to stop.'}
            {!isRecording && isTranscribing && '⏳ Converting your voice to text...'}
            {!isRecording && !isTranscribing && loading && '🤔 Thinking...'}
            {!isRecording && !isTranscribing && !loading && !isSpeaking && (
              <>
                Session: <strong>{sessionId.slice(0, 16)}...</strong>
              </>
            )}
            {isSpeaking && '🔊 Playing response...'}
          </div>
        </div>
      </div>
    </div>
  );
}
