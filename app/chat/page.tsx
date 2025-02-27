'use client';
import React, { useState, useRef, useEffect } from 'react';
import { useChat } from 'ai/react';
import { Settings, Plus, MessageCircle, FileText, Send, Menu, X, Loader, Users, Volume2, VolumeX, Mic, Square, Home } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'lottie-player': {
        ref?: any;
        src: string;
        background?: string;
        speed?: string;
        style?: React.CSSProperties;
        loop?: boolean;
        autoplay?: boolean;
      };
    }
  }
}

interface PersonaSelectorProps {
  selectedPersona: string;
  onPersonaChange: (persona: string) => void;
}

interface TTSControlsProps {
  messageContent: string;
  messageId: string;
  isEnabled: boolean;
}

interface VoiceRecorderProps {
  onTranscription: (text: string) => void;
  disabled?: boolean;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ onTranscription, disabled }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const lottiePlayerRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js';
    script.async = true;
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, []);

  const startRecording = async () => {
    try {
      if (!navigator.mediaDevices || !window.MediaRecorder) {
        throw new Error('Browser does not support voice recording. Please use Chrome, Firefox, or Edge.');
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        } 
      });
      
      streamRef.current = stream;

      const mimeTypes = ['audio/webm', 'audio/mp4', 'audio/ogg', 'audio/wav'];
      let mimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || '';
      
      if (!mimeType) {
        throw new Error('No supported audio format found');
      }

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: 128000
      });

      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        try {
          const audioBlob = new Blob(chunksRef.current, { type: mimeType });
          await processAudio(audioBlob);
        } catch (err) {
          console.error('Error processing audio:', err);
          setError('Failed to process audio');
        } finally {
          if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
          }
        }
      };

      mediaRecorderRef.current.start(100);
      setIsRecording(true);
      setError(null);

      if (lottiePlayerRef.current) {
        lottiePlayerRef.current.play();
      }
    } catch (err) {
      console.error('Error starting recording:', err);
      setError(err instanceof Error ? err.message : 'Failed to start recording');
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      try {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        if (lottiePlayerRef.current) {
          lottiePlayerRef.current.pause();
          lottiePlayerRef.current.currentTime = 0;
        }
      } catch (err) {
        console.error('Error stopping recording:', err);
        setError('Failed to stop recording');
      }
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob);

      const response = await fetch('/api/stt', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process audio');
      }

      const data = await response.json();
      if (data.text) {
        onTranscription(data.text);
      }
    } catch (err) {
      console.error('Error processing audio:', err);
      setError(err instanceof Error ? err.message : 'Failed to process audio');
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      setIsRecording(false);
      setIsProcessing(false);
    };
  }, []);

  return (
    <div className="flex items-center space-x-2">
      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={disabled || isProcessing}
        className={`p-2 rounded-full transition-all duration-200 ${
          isRecording ? 'bg-transparent scale-125' : 'bg-gray-100'
        } hover:bg-opacity-90 disabled:opacity-50`}
        title={isRecording ? 'Stop Recording' : 'Start Recording'}
        type="button"
      >
        {isProcessing ? (
          <Loader className="w-4 h-4 animate-spin text-gray-600" />
        ) : isRecording ? (
          <div className="w-12 h-12 transform scale-125">
            <video
              ref={lottiePlayerRef as any}
              src="/Animation - 1736917881376.webm"
              className="w-full h-full"
              loop
              autoPlay
              muted
            />
          </div>
        ) : (
          <Mic className="w-4 h-4 text-gray-600" />
        )}
      </button>
      
      {error && (
        <span className="text-xs text-red-500">{error}</span>
      )}
    </div>
  );
};

const TTSControls: React.FC<TTSControlsProps> = ({ messageContent, messageId, isEnabled }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  const togglePlayback = async () => {
    if (isLoading || !isEnabled) return;
    
    if (isPlaying) {
      audioRef.current?.pause();
      setIsPlaying(false);
      return;
    }
    
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: messageContent
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate audio');
      }
      
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
        setIsPlaying(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to play audio');
    } finally {
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        URL.revokeObjectURL(audioRef.current.src);
      }
    };
  }, []);
  
  if (!isEnabled) return null;
  
  return (
    <div className="flex items-center space-x-2">
      <button
        onClick={togglePlayback}
        disabled={isLoading}
        className={`p-2 rounded-full transition-colors ${
          isPlaying ? 'bg-[#4FD1C5]' : 'bg-gray-100'
        } hover:bg-[#45B8AE] disabled:opacity-50`}
        title={isPlaying ? 'Stop' : 'Play'}
      >
        {isLoading ? (
          <Loader className="w-4 h-4 animate-spin text-gray-600" />
        ) : isPlaying ? (
          <VolumeX className="w-4 h-4 text-white" />
        ) : (
          <Volume2 className="w-4 h-4 text-gray-600" />
        )}
      </button>
      
      {error && (
        <span className="text-xs text-red-500">Failed to play audio</span>
      )}
      
      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        onError={() => {
          setError('Audio playback failed');
          setIsPlaying(false);
        }}
      />
    </div>
  );
};

const PersonaSelector: React.FC<PersonaSelectorProps> = ({ selectedPersona, onPersonaChange }) => {
  return (
    <div className="flex items-center space-x-2 px-2">
      <Users size={20} className="text-gray-500" />
      <select
        value={selectedPersona}
        onChange={(e) => onPersonaChange(e.target.value)}
        className="text-sm border border-gray-200 rounded-md p-1 focus:outline-none focus:border-[#4FD1C5] focus:ring-1 focus:ring-[#4FD1C5]"
      >
        <option value="general">General</option>
        <option value="roleplay">Role Play</option>
      </select>
    </div>
  );
};

const LoadingSpinner = () => (
  <div className="flex items-center justify-center">
    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-[#4FD1C5]" />
  </div>
);

export default function ChatPage() {
  const [userId] = useState(() => uuidv4());
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showButtons, setShowButtons] = useState(true);
  const [isTTSEnabled, setIsTTSEnabled] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [isQuestionSelected, setIsQuestionSelected] = useState(false);
  const [selectedPersona] = useState('roleplay');
  const [instructionsShown, setInstructionsShown] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);

  const { messages, input, handleInputChange, handleSubmit, isLoading, reload, setMessages, setInput } = useChat({
    api: '/api/chat',
    body: { 
      userId,
      persona: selectedPersona,
      tts: isTTSEnabled,
      mode: 'chat'
    },
    onResponse: (response) => {
      console.log('Response started:', response);
      setError(null);
      setIsQuestionSelected(false);
    },
    onFinish: (message) => {
      if (message.content.trim()) {
        fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            traceResponse: true,
            userId,
            assistantResponse: message.content,
          }),
        });
      }
    },
    onError: (error) => {
      console.error('Chat error:', error);
      setError('Failed to process your request. Please try again.');
    }
  });

  const startScenario = async (type: 'Begin' | 'Instructions') => {
    const startMessage = type;
    setInput(startMessage);
    
    try {
      const fakeEvent = new Event('submit') as unknown as React.FormEvent<HTMLFormElement>;
      await handleSubmit(fakeEvent);
    } catch (error) {
      console.error(`Error starting ${type} scenario:`, error);
      setError(`Failed to start ${type} scenario`);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const logUserQuestion = async (question: string) => {
    try {
      const logData = {
        userId,
        timestamp: new Date().toISOString(),
        question,
      };

      await fetch('/api/logging', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(logData),
      });
    } catch (error) {
      console.error('Error logging question:', error);
    }
  };

  const enhancedSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    try {
      await logUserQuestion(input.trim());
      await handleSubmit(e);
    } catch (error) {
      console.error('Error submitting question:', error);
      setError('Failed to process your request');
    }
  };

  const handleNewChat = (e: React.MouseEvent) => {
    e.preventDefault();
    setMessages([]);
    setInput('');
    setShowButtons(true);
    setError(null);
  };

  const handleBeginClick = () => {
    setShowButtons(false);
    setInstructionsShown(false);
    setInput('Begin');
    setTimeout(() => {
      formRef.current?.requestSubmit();
    }, 0);
  };

  const handleInstructionsClick = () => {
    setShowButtons(false);
    setInstructionsShown(true);
    setInput('Instructions');
    setTimeout(() => {
      formRef.current?.requestSubmit();
    }, 0);
  };

  const handlePostInstructionsBegin = () => {
    setShowButtons(false);
    setInstructionsShown(false);
    setInput('Begin');
    setTimeout(() => {
      formRef.current?.requestSubmit();
    }, 0);
  };

  const customHandleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleInputChange(e);
    setIsQuestionSelected(!!e.target.value.trim());
  };

  return (
    <div className="flex h-screen bg-slate-50 relative">
      {/* Mobile Menu Button */}
      <button 
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        className="lg:hidden absolute top-4 left-4 z-50 text-slate-600"
      >
        {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Sidebar */}
      <div className={`
        fixed lg:static inset-y-0 left-0 z-40
        w-64 bg-gray-100 text-gray-800 transform transition-transform duration-200 ease-in-out
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="flex flex-col h-full">
          <div className="p-4">
            <Link href="/" className="flex items-center space-x-2 mb-8">
              <img src="Side-text.png" alt="Logo" className="w-30 h-10" />
            </Link>
            
            <div className="mb-8">
              <div className="flex items-center space-x-2 mb-4">
                <MessageCircle size={20} className="text-gray-700" />
                <span className="text-gray-700 font-medium">Recent Chats</span>
              </div>
              <div className="text-sm">
                {messages.length > 0 && (
                  <div className="p-2 hover:bg-gray-200 rounded transition-colors cursor-pointer">
                    Last conversation
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col w-full">
        {/* Header */}
        <div className="bg-white p-4 flex justify-between items-center border-b border-slate-200">
          <h2 className="text-xl text-[#2D3748] ml-12 lg:ml-0">Teach-Back: Drug Pricing Analogy</h2>
          <div className="flex space-x-2 sm:space-x-4">
            <Link 
              href="/"
              className="bg-[#4FD1C5] text-white px-2 sm:px-4 py-2 rounded-md flex items-center space-x-1 sm:space-x-2 hover:bg-[#45B8AE] transition-colors">
              <Home size={20} />
              <span className="hidden sm:inline">Home</span>
            </Link>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-auto p-2 sm:p-4 bg-slate-50">
          {/* Welcome Message */}
          {messages.length === 0 && (
            <div className="text-center my-4 sm:my-8">
              <h2 className="text-xl sm:text-2xl font-bold mb-4 text-[#2D3748]">👋 Hi There!</h2>
              <p className="text-slate-600 mb-8">
              You will practice articulating how drug pricing benchmarks impact pharmacy costs and reimbursement, and why this matters in pharmacy benefits consulting (Standard 2.4.1). To start, click "Instructions" or "Begin" 
              </p>
              
              <div className="max-w-2xl mx-auto px-2 sm:px-0">
                {/* Main Action Buttons */}
                {showButtons && (
                  <div className="flex justify-center space-x-6 mb-8">
                    <button 
                      onClick={handleBeginClick}
                      className="bg-[#4FD1C5] text-white px-2 sm:px-4 py-2 rounded-md flex items-center space-x-1 sm:space-x-2 hover:bg-[#45B8AE] transition-colors"
                    >
                      <FileText size={20} />
                      <span className="hidden sm:inline">Begin</span>
                    </button>
                    <button 
                      onClick={handleInstructionsClick}
                      className="bg-[#4FD1C5] text-white px-2 sm:px-4 py-2 rounded-md flex items-center space-x-1 sm:space-x-2 hover:bg-[#45B8AE] transition-colors"
                    >
                      <FileText size={20} />
                      <span className="hidden sm:inline">Instructions</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Chat Messages */}
          <div className="space-y-4">
            {messages.map(m => (
              <div
                key={m.id}
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {m.role === 'assistant' && (
                  <div className="mr-2 flex items-start pt-2">
                    <div className="w-8 h-8 rounded-full bg-[#4FD1C5] flex items-center justify-center">
                      <MessageCircle size={20} className="text-white" />
                    </div>
                  </div>
                )}
                <div
                  className={`max-w-[85%] sm:max-w-[75%] p-3 sm:p-4 rounded-lg shadow-sm ${
                    m.role === 'user'
                      ? 'bg-[#E56B8C] text-white'
                      : 'bg-white border border-slate-200'
                  }`}
                >
                  <div className="text-sm sm:text-base prose prose-slate max-w-none">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                  {m.role === 'assistant' && (
                    <div className="mt-2">
                      <TTSControls 
                        messageContent={m.content} 
                        messageId={m.id} 
                        isEnabled={isTTSEnabled}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {/* Add Begin button after instructions */}
            {instructionsShown && messages.length > 0 && messages[messages.length - 1].role === 'assistant' && (
              <div className="flex justify-start ml-10">
                <button 
                  onClick={handlePostInstructionsBegin}
                  className="bg-[#4FD1C5] text-white px-4 py-2 rounded-md flex items-center space-x-2 hover:bg-[#45B8AE] transition-colors"
                >
                  <FileText size={20} />
                  <span>Begin</span>
                </button>
              </div>
            )}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="mr-2 flex items-start pt-2">
                  <div className="w-8 h-8 rounded-full bg-[#4FD1C5] flex items-center justify-center">
                    <MessageCircle size={20} className="text-white" />
                  </div>
                </div>
                <div className="bg-white border border-slate-200 p-4 rounded-lg shadow-sm">
                  <LoadingSpinner />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area with Persona Selector and Voice Input */}
        <div className="p-2 sm:p-4 border-t border-slate-200 bg-white">
          <form ref={formRef} onSubmit={enhancedSubmit} className="space-y-2">
            <div className="flex space-x-2 sm:space-x-4">
              <input
                value={input}
                onChange={customHandleInputChange}
                placeholder="Type your message here..."
                className="flex-1 p-2 sm:p-3 text-sm sm:text-base border border-slate-200 rounded-md focus:outline-none focus:border-[#4FD1C5] focus:ring-1 focus:ring-[#4FD1C5] bg-white text-slate-900"
              />
              <VoiceRecorder
                onTranscription={(text) => {
                  setInput(text);
                  setIsQuestionSelected(true);
                }}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || (!input.trim() && !isQuestionSelected)}
                className={`bg-[#E56B8C] text-white px-3 sm:px-4 py-2 rounded-md flex items-center space-x-1 sm:space-x-2 hover:bg-[#D15A7B] transition-colors ${
                  (!input.trim() && !isQuestionSelected) ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isLoading ? (
                  <LoadingSpinner />
                ) : (
                  <>
                    <Send size={20} />
                    <span className="hidden sm:inline">Send</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Overlay for mobile sidebar */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Error Message */}
      {error && (
        <div className="fixed bottom-20 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <span className="block sm:inline">{error}</span>
          <button
            onClick={() => setError(null)}
            className="absolute top-0 bottom-0 right-0 px-4"
          >
            <span className="text-red-500">&times;</span>
          </button>
        </div>
      )}
    </div>
  );
}
