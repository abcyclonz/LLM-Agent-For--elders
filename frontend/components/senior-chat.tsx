"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import {
  Send,
  Heart,
  Calendar,
  Brain,
  User,
  FileText,
  RotateCcw,
  ChevronDown,
  ChevronRight,
  Settings,
  Database,
  Activity,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  X,
  Play,
  Pause,
} from "lucide-react"
import { useTheme } from "next-themes"
import { ThemeToggle } from "./theme-toggle"
import Image from "next/image"
import { useAuth } from "@/components/auth-context"
// WaveSurfer is only used for displaying the *static* waveform of a message, not live
// import WaveSurfer from "wavesurfer.js" // Not used in this revised version, but kept if needed for other purposes
import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  audioBlob?: Blob
}

export function SeniorChat() {
  const { theme } = useTheme()
  const { logout, sessionId } = useAuth();
  const [mounted, setMounted] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const [agentState, setAgentState] = useState({
    sessionId: "frontend-session",
    userToken: typeof window !== 'undefined' ? localStorage.getItem('memora_token') : null,
    userPersona: null, // Will be fetched from backend
    turnCount: 0,
    routerDecision: "",
    retrievedContext: "",
    toolResult: null,
    healthAlerts: null,
  })

  const [expandedSections, setExpandedSections] = useState({
    persona: true,
    context: false,
    facts: false,
    summaries: false,
  })

  const [activeSubTab, setActiveSubTab] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [isProcessingAudio, setIsProcessingAudio] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [showRecordingPulse, setShowRecordingPulse] = useState(false)
  const [showTranscriptionReady, setShowTranscriptionReady] = useState(false)
  const [previewMessageId, setPreviewMessageId] = useState<string | null>(null)

  const [playingAudioUrl, setPlayingAudioUrl] = useState<string | null>(null)
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const synthRef = useRef<any>(null)
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null)
  const pulseTimerRef = useRef<NodeJS.Timeout | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const [recordingAudioBlob, setRecordingAudioBlob] = useState<Blob | null>(null)
  const [showTextForMessages, setShowTextForMessages] = useState<Set<string>>(new Set())

  // --- Add state for profile and memory data ---
  const [profileData, setProfileData] = useState<any | null>(null)
  const [profileLoading, setProfileLoading] = useState(false)
  const [profileError, setProfileError] = useState<string | null>(null)
  const [facts, setFacts] = useState<any[]>([])
  const [summaries, setSummaries] = useState<any[]>([])
  const [memoryLoading, setMemoryLoading] = useState(false)
  const [memoryError, setMemoryError] = useState<string | null>(null)

  // --- Add state for health data ---
  const [healthData, setHealthData] = useState<any | null>(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [healthError, setHealthError] = useState<string | null>(null)

  const { userToken, userPersona, sessionId: agentSessionId } = agentState;

  // Audio feedback functions - Apple-like arpeggiated chimes, slightly lower pitch
  const playStartSound = () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      }
      const now = audioContextRef.current.currentTime
      // Upward arpeggio: Bb3 -> D4
      const osc1 = audioContextRef.current.createOscillator()
      const osc2 = audioContextRef.current.createOscillator()
      const gain1 = audioContextRef.current.createGain()
      const gain2 = audioContextRef.current.createGain()
      osc1.type = 'sine'; osc2.type = 'sine';
      osc1.connect(gain1); osc2.connect(gain2);
      gain1.connect(audioContextRef.current.destination)
      gain2.connect(audioContextRef.current.destination)
      // Bb3
      osc1.frequency.setValueAtTime(233.08, now)
      gain1.gain.setValueAtTime(0, now)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.01)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.09)
      gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.18)
      osc1.start(now); osc1.stop(now + 0.18);
      // D4
      osc2.frequency.setValueAtTime(293.66, now + 0.08)
      gain2.gain.setValueAtTime(0, now + 0.08)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.09)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.15)
      gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.23)
      osc2.start(now + 0.08); osc2.stop(now + 0.23);
    } catch (error) { /* Audio feedback not supported */ }
  }

  const playStopSound = () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      }
      const now = audioContextRef.current.currentTime
      // Downward arpeggio: D4 -> Bb3
      const osc1 = audioContextRef.current.createOscillator()
      const osc2 = audioContextRef.current.createOscillator()
      const gain1 = audioContextRef.current.createGain()
      const gain2 = audioContextRef.current.createGain()
      osc1.type = 'sine'; osc2.type = 'sine';
      osc1.connect(gain1); osc2.connect(gain2);
      gain1.connect(audioContextRef.current.destination)
      gain2.connect(audioContextRef.current.destination)
      // D4
      osc1.frequency.setValueAtTime(293.66, now)
      gain1.gain.setValueAtTime(0, now)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.01)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.09)
      gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.18)
      osc1.start(now); osc1.stop(now + 0.18);
      // Bb3
      osc2.frequency.setValueAtTime(233.08, now + 0.08)
      gain2.gain.setValueAtTime(0, now + 0.08)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.09)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.15)
      gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.23)
      osc2.start(now + 0.08); osc2.stop(now + 0.23);
    } catch (error) { /* Audio feedback not supported */ }
  }

  const playSendSound = () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      }
      const now = audioContextRef.current.currentTime
      // Quick two-note chime: F4 -> Bb4
      const osc1 = audioContextRef.current.createOscillator()
      const osc2 = audioContextRef.current.createOscillator()
      const gain1 = audioContextRef.current.createGain()
      const gain2 = audioContextRef.current.createGain()
      osc1.type = 'sine'; osc2.type = 'sine';
      osc1.connect(gain1); osc2.connect(gain2);
      gain1.connect(audioContextRef.current.destination)
      gain2.connect(audioContextRef.current.destination)
      // F4
      osc1.frequency.setValueAtTime(349.23, now)
      gain1.gain.setValueAtTime(0, now)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.01)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.09)
      gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.18)
      osc1.start(now); osc1.stop(now + 0.18);
      // Bb4
      osc2.frequency.setValueAtTime(466.16, now + 0.08)
      gain2.gain.setValueAtTime(0, now + 0.08)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.09)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.15)
      gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.23)
      osc2.start(now + 0.08); osc2.stop(now + 0.23);
    } catch (error) { /* Audio feedback not supported */ }
  }

  const playTranscriptionSound = () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      }
      const now = audioContextRef.current.currentTime
      // Slightly lower arpeggio: Bb3, D4, F4
      const osc1 = audioContextRef.current.createOscillator()
      const osc2 = audioContextRef.current.createOscillator()
      const osc3 = audioContextRef.current.createOscillator()
      const gain1 = audioContextRef.current.createGain()
      const gain2 = audioContextRef.current.createGain()
      const gain3 = audioContextRef.current.createGain()
      osc1.type = 'sine'; osc2.type = 'sine'; osc3.type = 'sine';
      osc1.connect(gain1); osc2.connect(gain2); osc3.connect(gain3);
      gain1.connect(audioContextRef.current.destination)
      gain2.connect(audioContextRef.current.destination)
      gain3.connect(audioContextRef.current.destination)
      // Bb3
      osc1.frequency.setValueAtTime(233.08, now)
      gain1.gain.setValueAtTime(0, now)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.01)
      gain1.gain.linearRampToValueAtTime(0.4, now + 0.09)
      gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.18)
      osc1.start(now); osc1.stop(now + 0.18);
      // D4
      osc2.frequency.setValueAtTime(293.66, now + 0.08)
      gain2.gain.setValueAtTime(0, now + 0.08)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.09)
      gain2.gain.linearRampToValueAtTime(0.3, now + 0.15)
      gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.23)
      osc2.start(now + 0.08); osc2.stop(now + 0.23);
      // F4
      osc3.frequency.setValueAtTime(349.23, now + 0.16)
      gain3.gain.setValueAtTime(0, now + 0.16)
      gain3.gain.linearRampToValueAtTime(0.2, now + 0.17)
      gain3.gain.linearRampToValueAtTime(0.2, now + 0.22)
      gain3.gain.exponentialRampToValueAtTime(0.01, now + 0.28)
      osc3.start(now + 0.16); osc3.stop(now + 0.28);
    } catch (error) { /* Audio feedback not supported */ }
  }

  // Custom chat handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    playSendSound()
    setShowTranscriptionReady(false) // Clear the ready indicator

    let userMessage: Message

    // If there's a preview message, replace it instead of adding a new one
    if (previewMessageId) {
      userMessage = {
        id: previewMessageId,
        role: 'user',
        content: input.trim()
      }
      // Update the existing preview message
      setMessages(prev => prev.map(msg =>
        msg.id === previewMessageId ? userMessage : msg
      ))
      setPreviewMessageId(null) // Clear preview message ID
    } else {
      // Normal case - add new message
      userMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: input.trim()
      }
      setMessages(prev => [...prev, userMessage])
    }

    setInput("")
    setIsLoading(true)

    // --- PATCH: Always get latest sessionId and userToken from localStorage/AuthContext ---
    const latestSessionId = sessionId || (typeof window !== 'undefined' ? localStorage.getItem('memora_session_id') : null)
    const latestUserToken = typeof window !== 'undefined' ? localStorage.getItem('memora_token') : null
    const patchedAgentState = {
      ...agentState,
      sessionId: latestSessionId,
      userToken: latestUserToken
    }
    console.log("[DEBUG] Sending chat with agentState:", patchedAgentState)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages.filter(msg => msg.id !== previewMessageId), userMessage],
          agentState: patchedAgentState
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      console.log("API response:", data)

      if (data.error) {
        throw new Error(data.error)
      }

      // Add assistant message to chat
      let audioBlob: Blob | undefined = undefined;
      try {
        const ttsResponse = await fetch('http://localhost:8000/speak_response', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: agentState.sessionId,
            text_to_speak: data.message.content,
          }),
        });
        if (ttsResponse.ok) {
          audioBlob = await ttsResponse.blob();
        }
      } catch (e) {
        // Optionally handle TTS fetch error
        console.error("TTS fetch error:", e);
      }
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: data.message.content,
        audioBlob,
      };
      setMessages(prev => [...prev, assistantMessage])

      // Update agent state
      if (data.agentState) {
        setAgentState(data.agentState)
      }

      // Auto-speak the AI response
      if (data.message.content && speechSupported) {
        speakText(data.message.content)
      }

    } catch (error) {
      console.error("Chat error:", error)
      // Add error message to chat
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }

  // Initialize speech synthesis and check microphone support
  useEffect(() => {
    if (typeof window !== "undefined") {
      // Check for speech synthesis support
      if (window.speechSynthesis) {
        synthRef.current = window.speechSynthesis
      }

      // Check for microphone access support
      if (navigator.mediaDevices) {
        setSpeechSupported(true)
      }
    }
  }, [])

  useEffect(() => {
    setMounted(true)
  }, [])

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current)
      }
      if (pulseTimerRef.current) {
        clearInterval(pulseTimerRef.current)
      }
    }
  }, [])

  // Update agent state when user token changes
  useEffect(() => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('memora_token') : null
    setAgentState(prev => ({
      ...prev,
      userToken: token
    }))
  }, [])

  // Sync agentState.sessionId with AuthContext.sessionId
  useEffect(() => {
    if (sessionId) {
      setAgentState(prev => ({
        ...prev,
        sessionId
      }));
    }
  }, [sessionId]);

  // Voice recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorderRef.current = new MediaRecorder(stream)
      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
          // For live waveform: update the blob with the latest chunk
          setRecordingAudioBlob(new Blob(audioChunksRef.current, { type: 'audio/wav' }))
        }
      }

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        setRecordingAudioBlob(audioBlob)
        await sendAudioToBackend(audioBlob)
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop())
        setRecordingAudioBlob(null)
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)
      setRecordingDuration(0)
      setShowRecordingPulse(true)
      playStartSound()

      // Start recording timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1)
      }, 1000)

      // Start pulse animation
      pulseTimerRef.current = setInterval(() => {
        setShowRecordingPulse(prev => !prev)
      }, 500)

    } catch (error) {
      console.error("Error starting recording:", error)
      alert("Could not access microphone. Please check permissions.")
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      setShowRecordingPulse(false)
      playStopSound()

      // Clear timers
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current)
        recordingTimerRef.current = null
      }
      if (pulseTimerRef.current) {
        clearInterval(pulseTimerRef.current)
        pulseTimerRef.current = null
      }
    }
  }

  const sendAudioToBackend = async (audioBlob: Blob) => {
    setIsProcessingAudio(true)

    try {
      const formData = new FormData()
      formData.append('audio_file', audioBlob, 'recording.wav')

      const response = await fetch('http://localhost:8000/transcribe_audio', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      console.log("Transcription API response:", data)

      if (data.error) {
        throw new Error(data.error)
      }

      // Show transcribed text and play notification sound
      if (data.transcribed_text) {
        playTranscriptionSound()
        setInput(data.transcribed_text)
        setShowTranscriptionReady(true)
        setTimeout(() => {
          setShowTranscriptionReady(false)
        }, 5000)
      }
    } catch (error) {
      console.error("Voice transcription error:", error)
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error transcribing your voice: ${error instanceof Error ? error.message : 'Unknown error'}`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsProcessingAudio(false)
    }
  }

  // --- Play backend-generated TTS audio ---
  const speakText = async (text: string, messageId?: string) => {
    if (isSpeaking) {
      stopSpeaking()
      return
    }
    setIsSpeaking(true)
    setPlayingMessageId(messageId || null)
    try {
      // Fetch TTS audio from backend
      const response = await fetch('http://localhost:8000/speak_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: agentState.sessionId,
          text_to_speak: text,
        }),
      })
      if (!response.ok) throw new Error('Failed to fetch TTS audio')
      const audioBlob = await response.blob()
      const audioUrl = URL.createObjectURL(audioBlob)
      setPlayingAudioUrl(audioUrl)
      // Play audio
      setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.play()
        }
      }, 100) // slight delay to ensure ref is set
    } catch (err) {
      setIsSpeaking(false)
      setPlayingMessageId(null)
      setPlayingAudioUrl(null)
      alert('Could not play audio response.')
    }
  }

  // --- Stop TTS audio playback ---
  const stopSpeaking = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    setIsSpeaking(false)
    setPlayingMessageId(null)
    if (playingAudioUrl) {
      URL.revokeObjectURL(playingAudioUrl)
      setPlayingAudioUrl(null)
    }
  }

  // --- Cleanup audio URL on unmount or when new audio is played ---
  useEffect(() => {
    return () => {
      if (playingAudioUrl) {
        URL.revokeObjectURL(playingAudioUrl)
      }
    }
  }, [playingAudioUrl])

  const handleReset = () => {
    setAgentState({
      sessionId: "frontend-session",
      userToken: null,
      userPersona: null, // Reset persona
      turnCount: 0,
      routerDecision: "",
      retrievedContext: "",
      toolResult: null,
      healthAlerts: null,
    })
    window.location.reload()
  }

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }))
  }

  const getActionIcon = (decision: string) => {
    switch (decision) {
      case "USE_CALENDAR_TOOL":
        return <Calendar className="h-3 w-3" />
      case "RETRIEVE_MEMORY":
        return <Brain className="h-3 w-3" />
      default:
        return <Heart className="h-3 w-3" />
    }
  }

  const getActionLabel = (decision: string) => {
    switch (decision) {
      case "USE_CALENDAR_TOOL":
        return "Calendar"
      case "RETRIEVE_MEMORY":
        return "Memory"
      default:
        return "Chat"
    }
  }

  // --- Fetch profile and memory data when subtab is opened ---
  useEffect(() => {
    if (!sessionId) return;
    if (activeSubTab === "profile") {
      setProfileLoading(true)
      setProfileError(null)
      fetch(`http://localhost:8000/get_profile/${sessionId}`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          return res.json()
        })
        .then(data => setProfileData(data.user_persona))
        .catch(e => setProfileError(e.message))
        .finally(() => setProfileLoading(false))
    }
    if (activeSubTab === "memory") {
      setMemoryLoading(true)
      setMemoryError(null)
      fetch(`http://localhost:8000/get_memories/${sessionId}`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          return res.json()
        })
        .then(data => {
          setFacts(data.facts || [])
          setSummaries(data.summaries || [])
        })
        .catch(e => setMemoryError(e.message))
        .finally(() => setMemoryLoading(false))
    }
    if (activeSubTab === "health") {
      setHealthLoading(true)
      setHealthError(null)
      fetch(`http://localhost:8000/get_health_data`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          return res.json()
        })
        .then(data => setHealthData(data))
        .catch(e => setHealthError(e.message))
        .finally(() => setHealthLoading(false))
    }
  }, [activeSubTab, sessionId])

  // Fetch user persona after login
  useEffect(() => {
    if (userToken && !userPersona && agentSessionId) {
      fetch(`http://localhost:8000/get_profile/${agentSessionId}`, {
        headers: { Authorization: `Bearer ${userToken}` },
      })
        .then((res) => res.json())
        .then((data) => {
          const persona = data.user_persona || data;
          setAgentState((prev) => ({ ...prev, userPersona: persona }));
        })
        .catch((err) => {
          setAgentState((prev) => ({ ...prev, userPersona: null }));
        });
    }
  }, [userToken, userPersona, agentSessionId]);

  const closeSubTab = () => {
    setActiveSubTab(null)
  }

  const renderSubTabContent = () => {
    switch (activeSubTab) {
      case "profile":
        return (
          <div className="flex-1 overflow-y-auto p-3 space-y-3 animate-in fade-in-50 duration-500 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
            <div className="space-y-3">
              <div className="bg-muted/20 rounded-lg transition-all duration-300 hover:bg-muted/25 hover:shadow-sm">
                <Collapsible open={expandedSections.persona} onOpenChange={() => toggleSection("persona")}> 
                  <CollapsibleTrigger asChild>
                    <div className="cursor-pointer hover:bg-muted/30 transition-all duration-200 ease-in-out p-3 rounded-lg">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <User className="h-4 w-4 text-muted-foreground transition-colors duration-200" />
                          User Information
                        </div>
                        <div className="transition-transform duration-300 ease-in-out">
                          {expandedSections.persona ? (
                            <ChevronDown className="h-4 w-4 text-muted-foreground" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          )}
                        </div>
                      </div>
                    </div>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="transition-smooth data-[state=open]:animate-smooth-fade-in data-[state=closed]:animate-smooth-slide-down">
                    <div className="px-3 pb-3">
                      {profileLoading ? (
                        <div className="text-xs text-muted-foreground">Loading profile...</div>
                      ) : profileError ? (
                        <div className="text-xs text-red-500">Error: {profileError}</div>
                      ) : profileData ? (
                        <div className="space-y-2 text-xs">
                          {Object.entries(profileData)
                            .filter(([key]) => !['id', 'created_at', 'updated_at'].includes(key))
                            .map(([key, value]) => (
                              <div key={key} className="flex items-start gap-2 animate-smooth-slide-up">
                                <span className="font-semibold capitalize">{key.replace(/_/g, ' ')}:</span> {Array.isArray(value) ? value.join(", ") : String(value)}
                              </div>
                            ))}
                        </div>
                      ) : (
                        <div className="text-xs text-muted-foreground">No profile data available.</div>
                      )}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
              {/* Keep context section as before */}
              <div className="bg-muted/20 rounded-lg transition-all duration-300 hover:bg-muted/25 hover:shadow-sm">
                <Collapsible open={expandedSections.context} onOpenChange={() => toggleSection("context")}> 
                  <CollapsibleTrigger asChild>
                    <div className="cursor-pointer hover:bg-muted/30 transition-all duration-200 ease-in-out p-3 rounded-lg">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <Brain className="h-4 w-4 text-muted-foreground transition-colors duration-200" />
                          Current Context
                        </div>
                        <div className="transition-transform duration-300 ease-in-out">
                          {expandedSections.context ? (
                            <ChevronDown className="h-4 w-4 text-muted-foreground" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          )}
                        </div>
                      </div>
                    </div>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="transition-all duration-300 ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:slide-out-to-top-1 data-[state=open]:slide-in-from-top-1">
                    <div className="px-3 pb-3">
                      <div className="text-xs text-muted-foreground animate-in fade-in-50 duration-300">
                        {agentState?.retrievedContext || "No context retrieved for this turn"}
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            </div>
          </div>
        )
      case "memory":
        return (
          <div className="flex-1 overflow-y-auto p-3 space-y-3 animate-in fade-in-50 duration-500 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
            <div className="space-y-3">
              <div className="bg-muted/20 rounded-lg transition-all duration-300 hover:bg-muted/25 hover:shadow-sm">
                <Collapsible open={expandedSections.facts} onOpenChange={() => toggleSection("facts")}> 
                  <CollapsibleTrigger asChild>
                    <div className="cursor-pointer hover:bg-muted/30 transition-all duration-200 ease-in-out p-3 rounded-lg">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-muted-foreground transition-colors duration-200" />
                          Facts ({facts.length})
                        </div>
                        <div className="transition-transform duration-300 ease-in-out">
                          {expandedSections.facts ? (
                            <ChevronDown className="h-4 w-4 text-muted-foreground" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          )}
                        </div>
                      </div>
                    </div>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="transition-all duration-300 ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:slide-out-to-top-1 data-[state=open]:slide-in-from-top-1">
                    <div className="px-3 pb-3">
                      {memoryLoading ? (
                        <div className="text-xs text-muted-foreground">Loading facts...</div>
                      ) : memoryError ? (
                        <div className="text-xs text-red-500">Error: {memoryError}</div>
                      ) : facts.length > 0 ? (
                        <div className="space-y-2">
                          {facts.map((fact, index) => (
                            <div
                              key={index}
                              className="p-2 bg-muted/30 rounded text-xs flex items-start gap-2 transition-all duration-300 hover:bg-muted/40 hover:shadow-sm animate-in fade-in-50 slide-in-from-left-2"
                              style={{ animationDelay: `${index * 100}ms` }}
                            >
                              <FileText className="h-3 w-3 text-muted-foreground mt-0.5 flex-shrink-0" />
                              <div>
                                <strong>Fact {index + 1}:</strong> {fact.document}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-xs text-muted-foreground">No facts available.</div>
                      )}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
              <div className="bg-muted/20 rounded-lg transition-all duration-300 hover:bg-muted/25 hover:shadow-sm">
                <Collapsible open={expandedSections.summaries} onOpenChange={() => toggleSection("summaries")}> 
                  <CollapsibleTrigger asChild>
                    <div className="cursor-pointer hover:bg-muted/30 transition-all duration-200 ease-in-out p-3 rounded-lg">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <Database className="h-4 w-4 text-muted-foreground transition-colors duration-200" />
                          Summaries ({summaries.length})
                        </div>
                        <div className="transition-transform duration-300 ease-in-out">
                          {expandedSections.summaries ? (
                            <ChevronDown className="h-4 w-4 text-muted-foreground" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          )}
                        </div>
                      </div>
                    </div>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="transition-all duration-300 ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:slide-out-to-top-1 data-[state=open]:slide-in-from-top-1">
                    <div className="px-3 pb-3">
                      {memoryLoading ? (
                        <div className="text-xs text-muted-foreground">Loading summaries...</div>
                      ) : memoryError ? (
                        <div className="text-xs text-red-500">Error: {memoryError}</div>
                      ) : summaries.length > 0 ? (
                        <div className="space-y-2">
                          {summaries.map((summary, index) => (
                            <div
                              key={index}
                              className="p-2 bg-muted/30 rounded text-xs flex items-start gap-2 transition-all duration-300 hover:bg-muted/40 hover:shadow-sm animate-in fade-in-50 slide-in-from-left-2"
                              style={{ animationDelay: `${index * 100}ms` }}
                            >
                              <Database className="h-3 w-3 text-muted-foreground mt-0.5 flex-shrink-0" />
                              <div>
                                <strong>Summary {index + 1}:</strong> {summary.document}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-xs text-muted-foreground">No summaries available.</div>
                      )}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
              <div
                className="bg-muted/20 rounded-lg p-3 transition-all duration-300 hover:bg-muted/25 hover:shadow-sm animate-in fade-in-50 duration-300"
                style={{ animationDelay: "200ms" }}
              >
                <div className="text-sm flex items-center gap-2 mb-2">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  Session Memory
                </div>
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div
                    className="flex items-center gap-2 animate-in fade-in-50 slide-in-from-left-2 duration-300"
                    style={{ animationDelay: "300ms" }}
                  >
                    <FileText className="h-3 w-3" />
                    Facts added: {facts.length}
                  </div>
                  <div
                    className="flex items-center gap-2 animate-in fade-in-50 slide-in-from-left-2 duration-300"
                    style={{ animationDelay: "400ms" }}
                  >
                    <Database className="h-3 w-3" />
                    Summaries added: {summaries.length}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case "health":
        // Prepare chart data if available
        let chartData = null;
        let chartOptions = null;
        if (healthData) {
          const vitalLabels = [];
          const vitalValues = [];
          if (typeof healthData.heart_rate !== 'undefined') {
            vitalLabels.push('Heart Rate');
            vitalValues.push(healthData.heart_rate);
          }
          if (typeof healthData.spo2 !== 'undefined') {
            vitalLabels.push('SpO2');
            vitalValues.push(healthData.spo2);
          }
          if (typeof healthData.bp_systolic !== 'undefined' && typeof healthData.bp_diastolic !== 'undefined') {
            vitalLabels.push('BP Systolic');
            vitalValues.push(healthData.bp_systolic);
            vitalLabels.push('BP Diastolic');
            vitalValues.push(healthData.bp_diastolic);
          }
          if (typeof healthData.sleep_score !== 'undefined') {
            vitalLabels.push('Sleep Score');
            vitalValues.push(healthData.sleep_score);
          }
          if (vitalLabels.length > 0) {
            chartData = {
              labels: vitalLabels,
              datasets: [
                {
                  label: 'Health Metrics',
                  data: vitalValues,
                  backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(54, 162, 235, 0.5)',
                    'rgba(255, 206, 86, 0.5)',
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(153, 102, 255, 0.5)',
                  ],
                  borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                  ],
                  borderWidth: 1,
                },
              ],
            };
            chartOptions = {
              responsive: true,
              plugins: {
                legend: { display: false },
                title: { display: true, text: 'Health Dashboard' },
              },
            };
          }
        }
// Inside renderSubTabContent for case "health":
return (
  <div className="flex-1 overflow-y-auto p-3 space-y-3 animate-in fade-in-50 duration-500 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
    <div className="space-y-3">
      <div className="bg-muted/20 rounded-lg transition-all duration-300 hover:bg-muted/25 hover:shadow-sm p-4">
        <div className="text-sm flex items-center gap-2 mb-2">
          <span role="img" aria-label="Health">🩺</span>
          Health Dashboard
        </div>
        {healthLoading ? (
          <div className="text-xs text-muted-foreground">Loading health data...</div>
        ) : healthError ? (
          <div className="text-xs text-red-500">Error: {healthError}</div>
        ) : healthData ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
              {/* Heart Rate Card */}
              {typeof healthData.heart_rate !== 'undefined' && (
                <div className="bg-card text-card-foreground rounded-lg p-4 shadow-sm flex flex-col items-center justify-center">
                  <div className="font-semibold text-lg mb-2">Heart Rate</div>
                  <div className="text-4xl font-bold text-primary">{healthData.heart_rate} <span className="text-base text-muted-foreground">BPM</span></div>
                  <div className="w-full mt-4" style={{ height: '200px' }}> {/* Explicit height for chart container */}
                    <Bar
                      data={{
                        labels: ['Heart Rate'],
                        datasets: [{
                          label: 'BPM',
                          data: [healthData.heart_rate],
                          backgroundColor: 'rgba(255, 99, 132, 0.7)',
                          borderColor: 'rgba(255, 99, 132, 1)',
                          borderWidth: 1,
                        }],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false, // Important for fixed height container
                        plugins: { legend: { display: false } },
                        scales: { y: { beginAtZero: true, max: 150 } }, // Adjusted max for typical heart rates
                      }}
                    />
                  </div>
                </div>
              )}
              {/* SpO2 Card */}
              {typeof healthData.spo2 !== 'undefined' && (
                <div className="bg-card text-card-foreground rounded-lg p-4 shadow-sm flex flex-col items-center justify-center">
                  <div className="font-semibold text-lg mb-2">Blood Oxygen (SpO₂)</div>
                  <div className="text-4xl font-bold text-blue-500">{healthData.spo2}<span className="text-base text-muted-foreground">%</span></div>
                  <div className="w-full mt-4" style={{ height: '200px' }}> {/* Explicit height for chart container */}
                    <Pie
                      data={{
                        labels: ['SpO₂', 'Remaining'],
                        datasets: [{
                          data: [healthData.spo2, 100 - healthData.spo2],
                          backgroundColor: ['rgba(54, 162, 235, 0.7)', 'rgba(200, 200, 200, 0.2)'],
                          borderColor: ['rgba(54, 162, 235, 1)', 'rgba(200, 200, 200, 0.2)'],
                          borderWidth: 1,
                        }],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false, // Important for fixed height container
                        plugins: { legend: { display: true, position: 'bottom' } },
                      }}
                    />
                  </div>
                </div>
              )}
              {/* Blood Pressure Card (spanning two columns) */}
              {typeof healthData.bp_systolic !== 'undefined' && typeof healthData.bp_diastolic !== 'undefined' && (
                <div className="bg-card text-card-foreground rounded-lg p-4 shadow-sm col-span-1 md:col-span-2 flex flex-col items-center justify-center">
                  <div className="font-semibold text-lg mb-2">Blood Pressure</div>
                  <div className="text-4xl font-bold text-purple-600">{healthData.bp_systolic}<span className="text-xl text-muted-foreground">/{healthData.bp_diastolic} mmHg</span></div>
                  <div className="w-full mt-4" style={{ height: '200px' }}> {/* Explicit height for chart container */}
                    <Bar
                      data={{
                        labels: ['Systolic', 'Diastolic'],
                        datasets: [{
                          label: 'mmHg',
                          data: [healthData.bp_systolic, healthData.bp_diastolic],
                          backgroundColor: ['rgba(255, 206, 86, 0.7)', 'rgba(75, 192, 192, 0.7)'],
                          borderColor: ['rgba(255, 206, 86, 1)', 'rgba(75, 192, 192, 1)'],
                          borderWidth: 1,
                        }],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false, // Important for fixed height container
                        plugins: { legend: { display: false } },
                        scales: { y: { beginAtZero: true, max: 200 } },
                      }}
                    />
                  </div>
                </div>
              )}
              {/* Sleep Score Card */}
              {typeof healthData.sleep_score !== 'undefined' && (
                <div className="bg-card text-card-foreground rounded-lg p-4 shadow-sm flex flex-col items-center justify-center">
                  <div className="font-semibold text-lg mb-2">Sleep Score</div>
                  <div className="text-4xl font-bold text-green-600">{healthData.sleep_score}<span className="text-base text-muted-foreground">/100</span></div>
                  <div className="w-full mt-4" style={{ height: '200px' }}> {/* Explicit height for chart container */}
                    <Pie
                      data={{
                        labels: ['Sleep Score', 'Remaining'],
                        datasets: [{
                          data: [healthData.sleep_score, 100 - healthData.sleep_score],
                          backgroundColor: ['rgba(153, 102, 255, 0.7)', 'rgba(200, 200, 200, 0.2)'],
                          borderColor: ['rgba(153, 102, 255, 1)', 'rgba(200, 200, 200, 0.2)'],
                          borderWidth: 1,
                        }],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false, // Important for fixed height container
                        plugins: { legend: { display: true, position: 'bottom' } },
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
            {/* Other Health Data (if any, kept for completeness) */}
            <div className="space-y-2 text-xs">
              {Object.entries(healthData).map(([key, value]) => {
                if (["heart_rate", "spo2", "bp_systolic", "bp_diastolic", "sleep_score"].includes(key)) return null;
                return (
                  <div key={key} className="flex items-start gap-2 animate-smooth-slide-up">
                    <span className="font-semibold capitalize">{key.replace(/_/g, ' ')}:</span> {String(value)}
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="text-xs text-muted-foreground">No health data available.</div>
        )}
      </div>
    </div>
  </div>
);

      case "settings":
        return (
          <div className="flex-1 overflow-y-auto p-3 space-y-3 animate-in fade-in-50 duration-500 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
            <div className="space-y-3">
              <div className="bg-muted/20 rounded-lg p-3 transition-all duration-300 hover:bg-muted/25 hover:shadow-sm animate-in fade-in-50 duration-300">
                <div className="text-sm flex items-center gap-2 mb-3">
                  <RotateCcw className="h-4 w-4 text-muted-foreground" />
                  Agent Controls
                </div>
                <div className="space-y-3">
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    className="w-full text-xs h-8 bg-muted/20 hover:bg-muted/40 border-muted transition-all duration-200 hover:shadow-sm hover:scale-[1.02] active:scale-[0.98]"
                    size="sm"
                  >
                    <RotateCcw className="h-3 w-3 mr-2 transition-transform duration-200 group-hover:rotate-180" />
                    Reset Chat & Memory
                  </Button>
                  <div
                    className="text-xs text-muted-foreground animate-in fade-in-50 duration-300"
                    style={{ animationDelay: "100ms" }}
                  >
                    This will clear all conversation history and reset the agent state.
                  </div>
                </div>
              </div>

              <div
                className="bg-muted/20 rounded-lg p-3 transition-all duration-300 hover:bg-muted/25 hover:shadow-sm animate-in fade-in-50 duration-300"
                style={{ animationDelay: "100ms" }}
              >
                <div className="text-sm flex items-center gap-2 mb-3">
                  <Settings className="h-4 w-4 text-muted-foreground" />
                  Preferences
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs">Theme</span>
                    <div className="transition-all duration-200 hover:scale-105">
                      <ThemeToggle />
                    </div>
                  </div>
                  <div
                    className="text-xs text-muted-foreground animate-in fade-in-50 duration-300"
                    style={{ animationDelay: "200ms" }}
                  >
                    Toggle between light and dark mode.
                  </div>
                </div>
              </div>

              {speechSupported && (
                <div
                  className="bg-muted/20 rounded-lg p-3 transition-all duration-300 hover:bg-muted/25 hover:shadow-sm animate-in fade-in-50 duration-300"
                  style={{ animationDelay: "200ms" }}
                >
                  <div className="text-sm flex items-center gap-2 mb-3">
                    <Volume2 className="h-4 w-4 text-muted-foreground" />
                    Voice Features
                  </div>
                  <div className="space-y-2 text-xs text-muted-foreground">
                    <div
                      className="flex items-center gap-2 animate-in fade-in-50 slide-in-from-left-2 duration-300"
                      style={{ animationDelay: "300ms" }}
                    >
                      <Mic className="h-3 w-3" />
                      Voice input enabled
                    </div>
                    <div
                      className="flex items-center gap-2 animate-in fade-in-50 slide-in-from-left-2 duration-300"
                      style={{ animationDelay: "400ms" }}
                    >
                      <Volume2 className="h-3 w-3" />
                      Text-to-speech enabled
                    </div>
                  </div>
                </div>
              )}

              <div
                className="bg-muted/20 rounded-lg p-3 transition-all duration-300 hover:bg-muted/25 hover:shadow-sm animate-in fade-in-50 duration-300"
                style={{ animationDelay: "300ms" }}
              >
                <div className="text-sm flex items-center gap-2 mb-3">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  Agent Status
                </div>
                <div className="space-y-2 text-xs">
                  <div
                    className="flex justify-between items-center animate-in fade-in-50 duration-300"
                    style={{ animationDelay: "400ms" }}
                  >
                    <span className="flex items-center gap-2">
                      {getActionIcon(agentState.routerDecision)}
                      Last Action:
                    </span>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-muted/40 text-muted-foreground transition-all duration-200 hover:bg-muted/50"
                    >
                      {getActionLabel(agentState.routerDecision) || "None"}
                    </Badge>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  // Modified LiveWaveform component for dynamic playback animation
  function LiveWaveform({ isRecording, isPlaying }: { isRecording?: boolean, isPlaying?: boolean }) {
    const [bars, setBars] = useState<number[]>([])
    const animationRef = useRef<number | undefined>(undefined)

    useEffect(() => {
      const animate = () => {
        // Generate random bar heights for live animation
        const newBars = Array.from({ length: 20 }, () =>
          Math.random() * 0.8 + 0.2 // Random height between 0.2 and 1.0
        )
        setBars(newBars)
        animationRef.current = requestAnimationFrame(animate)
      }

      if (isRecording || isPlaying) {
        animate()
      } else {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
        // When not recording or playing, set bars to a static, less active state
        setBars(Array.from({ length: 20 }, () => Math.random() * 0.2 + 0.1)) // Smaller, static bars
      }

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
    }, [isRecording, isPlaying])

    return (
      <div className="flex items-center gap-1 h-8 px-2">
        {bars.map((height, index) => (
          <div
            key={index}
            className={`w-1 rounded-full transition-all duration-100 ${
              isRecording
                ? 'bg-red-500 animate-pulse'
                : isPlaying
                ? 'bg-blue-500 animate-pulse'
                : 'bg-gray-400'
            }`}
            style={{
              height: `${height * 100}%`,
              animationDelay: `${index * 50}ms`,
              animationDuration: '1s'
            }}
          />
        ))}
      </div>
    )
  }

  // Add function to toggle text display for a message
  const toggleTextDisplay = (messageId: string) => {
    setShowTextForMessages(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

  // Add play/pause state for each assistant message
  const [playingWaveformId, setPlayingWaveformId] = useState<string | null>(null)
  const waveformAudioRefs = useRef<{ [id: string]: HTMLAudioElement | null }>({})
  // Add state to store audio URLs for each message
  const [audioUrls, setAudioUrls] = useState<{ [id: string]: string }>({})

  return (
    <div className="flex flex-col h-screen bg-background transition-smooth">
      {/* Restored Header with subtabs and Sign Out */}
      <div className="bg-muted/10 p-4 transition-smooth">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6 animate-smooth-fade-in">
            <div className="relative w-8 h-8 flex-shrink-0">
              {mounted && (
                <Image
                  src={theme === "dark" ? "/memora-dark.png" : "/memora-light.png"}
                  alt="Memora Logo"
                  width={32}
                  height={32}
                  className="object-contain transition-smooth hover:scale-110"
                  priority
                />
              )}
            </div>
            <div>
              <h1 className="text-xl font-medium text-foreground transition-smooth">Memora</h1>
              <p className="text-sm text-muted-foreground transition-smooth">
                Your kind, patient, and empathetic AI companion
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 animate-smooth-fade-in">
            {/* Sub-tab buttons */}
            <div className="flex items-center gap-1">
              <Button
                variant={activeSubTab === "profile" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setActiveSubTab(activeSubTab === "profile" ? null : "profile")}
                className="h-8 px-3 text-xs bg-muted/20 hover:bg-muted/40 text-muted-foreground transition-smooth-fast hover:scale-105 active:scale-95 hover-smooth"
              >
                <User className="h-3 w-3 mr-1 transition-smooth-fast" />
                Profile
              </Button>
              <Button
                variant={activeSubTab === "memory" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setActiveSubTab(activeSubTab === "memory" ? null : "memory")}
                className="h-8 px-3 text-xs bg-muted/20 hover:bg-muted/40 text-muted-foreground transition-smooth-fast hover:scale-105 active:scale-95 hover-smooth"
              >
                <Database className="h-3 w-3 mr-1 transition-smooth-fast" />
                Memory
              </Button>
              <Button
                variant={activeSubTab === "health" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setActiveSubTab(activeSubTab === "health" ? null : "health")}
                className="h-8 px-3 text-xs bg-muted/20 hover:bg-muted/40 text-muted-foreground transition-smooth-fast hover:scale-105 active:scale-95 hover-smooth"
              >
                <span role="img" aria-label="Health" className="h-3 w-3 mr-1 transition-smooth-fast">🩺</span>
                Health
              </Button>
              <Button
                variant={activeSubTab === "settings" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setActiveSubTab(activeSubTab === "settings" ? null : "settings")}
                className="h-8 px-3 text-xs bg-muted/20 hover:bg-muted/40 text-muted-foreground transition-smooth-fast hover:scale-105 active:scale-95 hover-smooth"
              >
                <Settings className="h-3 w-3 mr-1 transition-smooth-fast" />
                Settings
              </Button>
            </div>
            {/* Theme toggle */}
            <div className="transition-smooth-fast hover:scale-105">
              <ThemeToggle />
            </div>
            {/* Sign Out button */}
            <Button onClick={logout} variant="outline" className="ml-2">
              Sign Out
            </Button>
          </div>
        </div>
      </div>
      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Main Content Wrapper - animates width */}
        <div className={`main-content-compress${activeSubTab ? ' main-content-compress-active' : ''} h-full flex flex-col w-full min-h-0`}>
          {/* Chat Interface - Always visible */}
          <div className="flex flex-col flex-1 w-full min-h-0 transition-smooth-slow">
            {/* Messages */}
            <div className="flex-1 min-h-0 overflow-y-auto flex flex-col items-center p-4">
              <div className="w-full max-w-2xl space-y-6">
                {messages.length === 0 && (
                  <div className="flex flex-col justify-center items-center min-h-[60vh] p-6 text-center max-w-2xl mx-auto bg-muted/10 rounded-lg transition-smooth hover:bg-muted/15 animate-smooth-fade-in hover:bg-muted/15 animate-smooth-fade-in hover-smooth">
                    <div className="space-y-4 w-full">
                      <div className="flex justify-center">
                        <div
                          className="relative w-16 h-16 animate-smooth-bounce"
                          style={{ animationDelay: "200ms" }}
                        >
                          {mounted && (
                          <Image
                            src={theme === "dark" ? "/memora-dark.png" : "/memora-light.png"}
                            alt="Memora Logo"
                            width={64}
                            height={64}
                              className="object-contain transition-smooth-slow animate-smooth-float"
                            priority
                          />
                          )}
                        </div>
                      </div>
                      <h2
                        className="text-lg font-medium text-foreground animate-smooth-slide-up"
                        style={{ animationDelay: "300ms" }}
                      >
                        Welcome to Memora
                      </h2>
                      <p
                        className="text-muted-foreground animate-smooth-slide-up"
                        style={{ animationDelay: "400ms" }}
                      >
                        I'm here to be your supportive and engaging conversational partner. You can type your message or use
                        voice recording to chat with me.
                      </p>
                      <div className="flex flex-wrap gap-2 justify-center">
                        {[
                          { icon: FileText, text: "History Discussions" },
                          { icon: Calendar, text: "Calendar Management" },
                          { icon: Brain, text: "Memory Assistance" },
                          { icon: Heart, text: "Friendly Conversation" },
                        ].map((item, index) => (
                          <Badge
                            key={item.text}
                            variant="outline"
                            className="flex items-center gap-1 bg-muted/20 text-muted-foreground transition-smooth hover:bg-muted/30 hover:scale-105 animate-smooth-slide-up hover-smooth"
                            style={{ animationDelay: `${500 + index * 100}ms` }}
                          >
                            <item.icon className="h-3 w-3" />
                            {item.text}
                          </Badge>
                        ))}
                      </div>
                      {speechSupported && (
                        <div
                          className="flex items-center justify-center gap-2 text-sm text-muted-foreground animate-smooth-slide-up"
                          style={{ animationDelay: "900ms" }}
                        >
                          <Mic className="h-4 w-4" />
                          <span>Voice recording and text-to-speech enabled</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {messages.map((message, index) => (
                  <div
                    key={message.id}
                    className={`flex w-full animate-smooth-slide-up`}
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div
                      className={`max-w-xl rounded-lg px-4 py-2 transition-smooth hover:shadow-sm hover-smooth ${
                        message.role === "user"
                          ? `ml-auto bg-primary/80 text-primary-foreground hover:bg-primary/90${message.id === previewMessageId ? " bg-green-500/80 text-white hover:bg-green-500/90 border-2 border-green-400/50 animate-pulse" : ""}`
                          : `mr-auto ${theme === 'light' ? 'bg-gray-100' : 'bg-muted/30'} text-foreground ${theme === 'light' ? 'hover:bg-gray-200' : 'hover:bg-muted/40'}`
                      }`}
                    >
                      {message.role === "user" && (
                        <>
                          <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                          {message.id === previewMessageId && (
                            <div className="mt-1 text-xs text-green-200 opacity-75">
                              ✏️ Edit and click send to process
                            </div>
                          )}
                        </>
                      )}
                      {message.role === "assistant" && (
                        <>
                          {showTextForMessages.has(message.id) ? (
                            // Show text content only
                            <>
                              <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                              <div className="flex items-center gap-2 mt-2">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => toggleTextDisplay(message.id)}
                                  className="h-6 px-2 text-xs hover:bg-background/30 text-muted-foreground transition-smooth-fast hover:scale-105 active:scale-95"
                                >
                                  <Volume2 className="h-3 w-3 mr-1" /> Show audio
                                </Button>
                              </div>
                            </>
                          ) : (
                            // Show waveform and audio controls
                            <>
                              {message.audioBlob && (
                                <div className="flex items-center gap-2 mt-2">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={async () => {
                                      try {
                                        if (playingWaveformId === message.id) {
                                          waveformAudioRefs.current[message.id]?.pause()
                                        } else {
                                          // Pause any other playing audio
                                          Object.keys(waveformAudioRefs.current).forEach(id => {
                                            if (id !== message.id) waveformAudioRefs.current[id]?.pause()
                                          })

                                          // Create URL if it doesn't exist
                                          if (!audioUrls[message.id]) {
                                            const url = URL.createObjectURL(message.audioBlob!)
                                            setAudioUrls(prev => ({ ...prev, [message.id]: url }))
                                          }

                                          // Play the audio
                                          const audioElement = waveformAudioRefs.current[message.id]
                                          if (audioElement) {
                                            await audioElement.play()
                                          }
                                        }
                                      } catch (error) {
                                        console.error('Error playing audio:', error)
                                      }
                                    }}
                                    className="h-8 w-8 p-0 hover:bg-background/30 text-muted-foreground transition-smooth-fast hover:scale-105 active:scale-95"
                                  >
                                    {playingWaveformId === message.id ? (
                                      <Pause className="h-4 w-4" />
                                    ) : (
                                      <Play className="h-4 w-4" />
                                    )}
                                  </Button>
                                  <LiveWaveform isPlaying={playingWaveformId === message.id} />
                                  {audioUrls[message.id] && (
                                    <audio
                                      ref={el => { waveformAudioRefs.current[message.id] = el }}
                                      src={audioUrls[message.id]}
                                      onPlay={() => setPlayingWaveformId(message.id)}
                                      onPause={() => setPlayingWaveformId(prev => (prev === message.id ? null : prev))}
                                      onEnded={() => setPlayingWaveformId(null)}
                                      onError={(e) => console.error('Audio error:', e)}
                                      style={{ display: 'none' }}
                                    />
                                  )}
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => toggleTextDisplay(message.id)}
                                    className="h-6 px-2 text-xs hover:bg-background/30 text-muted-foreground transition-smooth-fast hover:scale-105 active:scale-95"
                                  >
                                    <FileText className="h-3 w-3 mr-1" /> Show text
                                  </Button>
                                </div>
                              )}
                            </>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                ))}
                {isRecording && (
                  <div className="flex w-full animate-smooth-slide-up">
                    <div className={`max-w-xl rounded-lg px-4 py-2 transition-smooth hover:shadow-sm hover-smooth ml-auto bg-primary/80 text-primary-foreground hover:bg-primary/90`}>
                      <div className="flex items-center gap-3">
                        <div className="relative">
                          <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse"></div>
                          <div className="absolute inset-0 w-4 h-4 bg-red-400 rounded-full animate-ping opacity-75"></div>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-sm font-medium">Recording...</span>
                          <span className="text-xs opacity-80">
                            {Math.floor(recordingDuration / 60)}:{(recordingDuration % 60).toString().padStart(2, '0')}
                          </span>
                        </div>
                        <LiveWaveform isRecording={true} />
                      </div>
                    </div>
                  </div>
                )}
                {showTranscriptionReady && (
                  <div className="flex w-full animate-smooth-slide-up">
                    <div className="max-w-xl rounded-lg px-4 py-2 transition-smooth hover:shadow-sm hover:smooth ml-auto bg-green-500/10 text-green-900 dark:text-green-200 border border-green-400/30 shadow-green-200/20">
                      <div className="flex items-center gap-3">
                        <div className="relative">
                          <div className="w-4 h-4 bg-green-500 rounded-full animate-pulse"></div>
                          <div className="absolute inset-0 w-4 h-4 bg-green-400 rounded-full animate-ping opacity-75"></div>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-sm font-medium">Transcription ready!</span>
                          <span className="text-xs opacity-80">Click send to process your message</span>
                        </div>
                        <div className="flex space-x-1">
                          {[...Array(3)].map((_, i) => (
                            <div
                              key={i}
                              className="w-1 h-3 bg-green-500/60 rounded-full"
                              style={{
                                animation: `processingWave 1.2s ease-in-out infinite`,
                                animationDelay: `${i * 0.15}s`
                              }}
                            ></div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                {isLoading && (
                  <div className="flex w-full animate-smooth-slide-up">
                    <div className={`max-w-xl rounded-lg px-4 py-2 transition-smooth hover:shadow-sm hover-smooth mr-auto ${theme === 'light' ? 'bg-gray-100' : 'bg-muted/30'} text-foreground ${theme === 'light' ? 'hover:bg-gray-200' : 'hover:bg-muted/40'}`}>
                      <div className="flex items-center space-x-2">
                        <div className="animate-smooth-pulse flex space-x-1">
                          <div className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-smooth-bounce"></div>
                          <div
                            className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-smooth-bounce"
                            style={{ animationDelay: "0.1s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-smooth-bounce"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                        </div>
                        <span className="text-sm text-muted-foreground">Memora is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              {isProcessingAudio && (
                <div className="flex justify-center animate-smooth-scale">
                  <div className="bg-primary/10 rounded-lg px-4 py-3 flex items-center gap-3 transition-all duration-300 hover:bg-primary/15">
                    <div className="relative">
                      <div className="w-4 h-4 bg-blue-500 rounded-full animate-spin"></div>
                      <div className="absolute inset-0 w-4 h-4 border-2 border-blue-300 border-t-transparent rounded-full animate-spin" style={{ animationDuration: '1s' }}></div>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-primary/90">Processing audio...</span>
                      <span className="text-xs text-primary/70">Transcribing your voice</span>
                    </div>
                    <div className="flex space-x-1">
                      {[...Array(4)].map((_, i) => (
                        <div
                          key={i}
                          className="w-1 h-4 bg-blue-500/60 rounded-full"
                          style={{
                            animation: `processingWave 1.5s ease-in-out infinite`,
                            animationDelay: `${i * 0.2}s`
                          }}
                        ></div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {recordingAudioBlob && !isRecording && (
                <div className="my-2">
                  {/* This LiveWaveform is just for a visual indicator, not actual playback */}
                  <LiveWaveform isRecording={false} isPlaying={false} />
                </div>
              )}
            </div>

              {/* Input - sticky at bottom, outside scrollable area */}
              <div className="bg-background p-4 transition-smooth border-t border-gray-200 dark:border-muted/30 sticky bottom-0 w-full z-10">
                <form onSubmit={handleSubmit} className="flex items-center gap-2 bg-background rounded-full px-6 py-3 shadow-md w-full max-w-2xl mx-auto mb-4">
                  <div className="flex-1 relative">
                    <Input
                      value={input}
                      onChange={handleInputChange}
                      placeholder={isRecording ? "Recording..." : "Type your message or click the microphone to speak..."}
                      className={`flex-1 bg-transparent border-none focus:ring-0 rounded-full text-base pr-12 transition-smooth-fast ${
                        showTranscriptionReady
                          ? 'border-green-500/50 bg-green-500/5 shadow-lg shadow-green-500/20 animate-pulse'
                          : ''
                      }`}
                      disabled={isLoading || isRecording}
                    />
                    {speechSupported && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={isRecording ? stopRecording : startRecording}
                        className={`absolute right-1 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full hover:bg-muted/40 text-muted-foreground transition-all duration-300 hover:scale-110 active:scale-95 ${
                          isRecording
                            ? "text-red-500 bg-red-500/10 shadow-lg shadow-red-500/20 animate-pulse"
                            : "hover:shadow-md hover:shadow-primary/20"
                        }`}
                        disabled={isLoading}
                        title={isRecording ? "Stop recording" : "Start voice recording"}
                      >
                        <div className="relative">
                          {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
                          {isRecording && (
                            <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
                          )}
                        </div>
                      </Button>
                    )}
                  </div>
                  <Button
                    type="submit"
                    disabled={isLoading || !input.trim() || isRecording}
                    className={`rounded-full transition-smooth-fast hover:scale-105 active:scale-95 hover:shadow-sm disabled:opacity-50 disabled:cursor-not-allowed ${
                      showTranscriptionReady
                        ? 'bg-green-500/80 hover:bg-green-500/90 text-white shadow-lg shadow-green-500/30 animate-pulse'
                        : 'bg-primary/70 hover:bg-primary/80 text-primary-foreground'
                    }`}
                  >
                    <Send className="h-4 w-4 transition-smooth-fast group-hover:translate-x-0.5" />
                  </Button>
                </form>
              </div>
            </div>
          </div>

        {/* Sub-tab Panel - Flex sibling, compresses main content */}
        {activeSubTab && (
          <div className={`bg-muted/10 animate-subtab-slide-in-end flex flex-col h-full shadow-xl ${activeSubTab === 'health' ? 'w-[32rem] max-w-full' : 'w-80'}`}>
            {/* Sub-tab Header */}
            <div className="bg-muted/20 p-3 flex items-center justify-between transition-smooth">
              <div className="flex items-center gap-2 animate-smooth-fade-in">
                {activeSubTab === "profile" && <User className="h-4 w-4 text-muted-foreground" />}
                {activeSubTab === "memory" && <Database className="h-4 w-4 text-muted-foreground" />}
                {activeSubTab === "health" && <span role="img" aria-label="Health">🩺</span>}
                {activeSubTab === "settings" && <Settings className="h-4 w-4 text-muted-foreground" />}
                <h2 className="text-sm font-medium capitalize text-muted-foreground">{activeSubTab}</h2>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={closeSubTab}
                className="h-6 w-6 hover:bg-muted/40 text-muted-foreground transition-smooth-fast hover:scale-110 active:scale-90 animate-smooth-fade-in"
              >
                <X className="h-3 w-3" />
                <span className="sr-only">Close {activeSubTab}</span>
              </Button>
            </div>

            {/* Sub-tab Content */}
            {renderSubTabContent()}
          </div>
        )}
      </div>
    </div>
  )
}