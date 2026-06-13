import React, { useState, useEffect, useRef, useCallback } from 'react';

export interface VoiceWidgetProps {
  /** WebSocket server URL (e.g., wss://voice.example.com/ws) */
  serverUrl: string;
  /** API key for authentication */
  apiKey?: string;
  /** Enable continuous conversation mode */
  continuousMode?: boolean;
  /** Callback when transcript is received */
  onTranscript?: (text: string) => void;
  /** Callback when AI responds (full text) */
  onResponse?: (text: string) => void;
  /** Callback on error */
  onError?: (error: string) => void;
  /** Custom button style */
  buttonStyle?: React.CSSProperties;
  /** Custom container style */
  style?: React.CSSProperties;
  /** Button size in pixels */
  size?: number;
  /** Primary color */
  color?: string;
}

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

function float32ToBase64(float32Array: Float32Array): string {
  const bytes = new Uint8Array(float32Array.buffer, float32Array.byteOffset, float32Array.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decodePcm(base64Data: string) {
  const binaryString = atob(base64Data);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, len >> 1);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }
  return float32;
}

export function VoiceWidget({
  serverUrl,
  apiKey,
  continuousMode = false,
  onTranscript,
  onResponse,
  onError,
  buttonStyle,
  style,
  size = 80,
  color = '#c9a227',
}: VoiceWidgetProps) {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const scheduledSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const nextStartTimeRef = useRef(0);
  const continuousRef = useRef(continuousMode);
  continuousRef.current = continuousMode;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus('connecting');

    const url = apiKey
      ? `${serverUrl}?api_key=${encodeURIComponent(apiKey)}`
      : serverUrl;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setStatus('connected');
    };

    ws.onclose = (event: CloseEvent) => {
      setStatus('disconnected');
      if (event.code === 4001) onError?.('API key required');
      else if (event.code === 4002) onError?.('Invalid API key');
      else if (event.code === 4003) onError?.('Rate limit exceeded');
    };

    ws.onerror = () => {
      setStatus('error');
      onError?.('Connection failed');
    };

    ws.onmessage = (event: MessageEvent) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    wsRef.current = ws;
  }, [serverUrl, apiKey, onError]);

  const cancelAllAudio = useCallback(() => {
    scheduledSourcesRef.current.forEach(src => {
      try { src.stop(); } catch { /* already ended */ }
    });
    scheduledSourcesRef.current = [];
    nextStartTimeRef.current = 0;
  }, []);

  const scheduleAudio = useCallback((base64Data: string, sampleRate: number) => {
    try {
      const ctx = audioCtxRef.current;
      if (!ctx) return;
      const float32 = decodePcm(base64Data);

      let audioData = float32;
      if (sampleRate !== ctx.sampleRate) {
        const ratio = ctx.sampleRate / sampleRate;
        const newLength = Math.round(float32.length * ratio);
        audioData = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
          const srcIdx = i / ratio;
          const srcFloor = Math.floor(srcIdx);
          const srcCeil = Math.min(srcFloor + 1, float32.length - 1);
          const frac = srcIdx - srcFloor;
          audioData[i] = float32[srcFloor] * (1 - frac) + float32[srcCeil] * frac;
        }
        sampleRate = ctx.sampleRate;
      }

      const buffer = ctx.createBuffer(1, audioData.length, sampleRate);
      buffer.copyToChannel(audioData, 0);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      const now = ctx.currentTime;
      const startTime = Math.max(nextStartTimeRef.current, now);

      source.start(startTime);
      nextStartTimeRef.current = startTime + buffer.duration;

      scheduledSourcesRef.current.push(source);
      source.onended = () => {
        const idx = scheduledSourcesRef.current.indexOf(source);
        if (idx > -1) scheduledSourcesRef.current.splice(idx, 1);
      };
    } catch (e) {
      console.error('Audio scheduling error:', e);
    }
  }, []);

  const handleMessage = useCallback((msg: any) => {
    switch (msg.type) {
      case 'listening_started':
        setIsListening(true);
        break;
      case 'listening_stopped':
        setIsListening(false);
        break;
      case 'transcript':
        onTranscript?.(msg.text);
        break;
      case 'response_chunk':
        break;
      case 'audio_chunk':
        setIsSpeaking(true);
        scheduleAudio(msg.data, msg.sample_rate);
        break;
      case 'response_complete':
        onResponse?.(msg.text);
        setIsSpeaking(false);
        if (continuousRef.current) {
          setTimeout(() => startListening(), 1000);
        }
        break;
    }
  }, [onTranscript, onResponse, scheduleAudio]);

  const startListening = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connect();
      return;
    }

    cancelAllAudio();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });

      mediaStreamRef.current = stream;
      audioCtxRef.current = new AudioContext({ sampleRate: 16000 });

      const source = audioCtxRef.current.createMediaStreamSource(stream);
      const processor = audioCtxRef.current.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const audioData = e.inputBuffer.getChannelData(0);
          wsRef.current.send(JSON.stringify({ type: 'audio', data: float32ToBase64(audioData) }));
        }
      };

      source.connect(processor);
      processor.connect(audioCtxRef.current.destination);
      processorRef.current = processor;

      wsRef.current.send(JSON.stringify({ type: 'start_listening' }));

    } catch {
      onError?.('Microphone access denied');
    }
  }, [connect, onError, cancelAllAudio]);

  const stopListening = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }

    wsRef.current?.send(JSON.stringify({ type: 'stop_listening' }));
  }, []);

  useEffect(() => {
    connect();
    return () => {
      cancelAllAudio();
      wsRef.current?.close();
    };
  }, [connect, cancelAllAudio]);

  const handleMouseDown = () => {
    if (!continuousMode) startListening();
  };

  const handleMouseUp = () => {
    if (!continuousMode) stopListening();
  };

  const handleClick = () => {
    if (continuousMode) {
      if (isListening) stopListening();
      else startListening();
    }
  };

  const label = isListening ? 'mic' : isSpeaking ? 'speaker' : 'mic-off';
  const buttonColor = isListening ? color : isSpeaking ? '#4fc3f7' : color;

  return (
    <div style={{ textAlign: 'center', ...style }}>
      <button
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
        disabled={status !== 'connected'}
        aria-label={label}
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          border: 'none',
          background: buttonColor,
          color: 'white',
          fontSize: size / 5,
          fontWeight: 600,
          cursor: status === 'connected' ? 'pointer' : 'not-allowed',
          opacity: status === 'connected' ? 1 : 0.5,
          transition: 'all 0.2s',
          boxShadow: isListening ? `0 0 20px ${color}` : 'none',
          ...buttonStyle,
        }}
      >
        {isListening ? 'mic' : isSpeaking ? 'volume-up' : 'mic'}
      </button>
      <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
        {status === 'connecting' && 'Connecting...'}
        {status === 'connected' && (isListening ? 'Listening...' : continuousMode ? 'Tap to talk' : 'Hold to talk')}
        {status === 'disconnected' && 'Disconnected'}
        {status === 'error' && 'Connection error'}
      </div>
    </div>
  );
}

export default VoiceWidget;
