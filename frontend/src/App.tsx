import React, { useState } from 'react';
import axios from 'axios';
import ModelCard from './components/ModelCard';
import ConfidenceGraph from './components/ConfidenceGraph';
import ValidityGraph from './components/ValidityGraph';
import type { PredictionResponse, GlobalStats, ModelResult } from './types';

// API CONSTANTS
const API_URL = 'http://localhost:8000';

function App() {
  const [language, setLanguage] = useState<'ta' | 'en'>('ta');
  const [prefixMode, setPrefixMode] = useState(false); // Toggle for prefix filtering

  const [input, setInput] = useState('');
  const [sentence, setSentence] = useState('');
  const [started, setStarted] = useState(false);

  // Helper function to extract the partial word being typed (after last space)
  const extractPartialWord = (text: string): string => {
    const lastSpaceIndex = text.lastIndexOf(' ');
    if (lastSpaceIndex === -1) {
      return text.trim(); // Entire input is the partial word
    }
    return text.substring(lastSpaceIndex + 1);
  };

  /* Debounced Auto-Prediction Logic */
  const [loading, setLoading] = useState(false);
  const abortControllerRef = React.useRef<AbortController | null>(null);
  
  React.useEffect(() => {
    if (!started || !sentence.trim()) return;

    const timeoutId = setTimeout(() => {
      fetchPredictions(sentence);
    }, 600); // 600ms debounce

    return () => clearTimeout(timeoutId);
  }, [sentence, started]);

  const [results, setResults] = useState<PredictionResponse | null>(null);

  const [stats, setStats] = useState<GlobalStats>({
    bi_gru_conf: [],
    lstm_conf: [],
    bilstm_conf: [],
    bi_gru_valid: [],
    lstm_valid: [],
    bilstm_valid: [],
    labels: []
  });

  // Prediction History Timeline
  interface HistoryEvent {
    timestamp: Date;
    context: string;
    predictions: {
      bi_gru: string[];
      lstm: string[];
      bilstm: string[];
    };
    chosenWord: string;
    correctModels: string[]; // Which models predicted the chosen word
  }

  const [history, setHistory] = useState<HistoryEvent[]>([]);

  // Reset state when language changes
  const switchLanguage = (lang: 'ta' | 'en') => {
    if (lang === language) return;
    setLanguage(lang);
    setInput('');
    setSentence('');
    setStarted(false);
    setResults(null);
    setStats({
      bi_gru_conf: [],
      lstm_conf: [],
      bilstm_conf: [],
      bi_gru_valid: [],
      lstm_valid: [],
      bilstm_valid: [],
      labels: []
    });
  };

  const fetchPredictions = async (currentText: string) => {
    setLoading(true);
    
    // Cancel previous request if it's still in flight
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create new AbortController for this request
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    
    try {
      // Extract the partial word being typed (for prefix filtering)
      // Only use it if prefix mode is enabled
      const partialWord = prefixMode ? extractPartialWord(currentText) : '';

      const res = await axios.post(`${API_URL}/predict`, {
        current_sentence: currentText,
        language: language,
        partial_word: partialWord
      }, {
        signal: abortController.signal
      });
      
      // Only process response if this request wasn't aborted
      if (!abortController.signal.aborted) {
          setResults(res.data.data);

        // Update Graph Stats
        const data: PredictionResponse = res.data.data;
        setStats(prev => ({
          // For English, bi_gru is unused/empty, but we keep the arrays aligned
          bi_gru_conf: [...prev.bi_gru_conf, data.bi_gru?.top1_confidence || 0],
          lstm_conf: [...prev.lstm_conf, data.lstm.top1_confidence],
          bilstm_conf: [...prev.bilstm_conf, data.bilstm.top1_confidence],

          bi_gru_valid: [...prev.bi_gru_valid, data.bi_gru?.validity_score || 0],
          lstm_valid: [...prev.lstm_valid, data.lstm.validity_score],
          bilstm_valid: [...prev.bilstm_valid, data.bilstm.validity_score],

          labels: [...prev.labels, `Step ${prev.labels.length + 1}`]
        }));
      }

    } catch (err) {
      // Only show error if request wasn't aborted
      if (!abortController.signal.aborted) {
        console.error("Prediction Error:", err);
        alert("Failed to fetch predictions. Ensure backend is running.");
      }
    } finally {
      // Only clear loading if this request wasn't aborted
      if (!abortController.signal.aborted) {
        setLoading(false);
      }
    }
  };

  const handleStart = () => {
    if (!input.trim()) return;
    setSentence(input);
    setStarted(true);
    // fetchPredictions(input); // Triggered by useEffect
  };

  const handleWordClick = (word: string) => {
    // Track this selection in history
    if (results) {
      const biGruPreds = results.bi_gru?.predictions?.map(p => p.word) || [];
      const lstmPreds = results.lstm?.predictions?.map(p => p.word) || [];
      const bilstmPreds = results.bilstm?.predictions?.map(p => p.word) || [];

      // Determine which models predicted this word
      const correctModels: string[] = [];
      if (biGruPreds.includes(word)) correctModels.push('Bi-GRU');
      if (lstmPreds.includes(word)) correctModels.push('LSTM');
      if (bilstmPreds.includes(word)) correctModels.push('BiLSTM');

      const historyEvent: HistoryEvent = {
        timestamp: new Date(),
        context: sentence,
        predictions: {
          bi_gru: biGruPreds,
          lstm: lstmPreds,
          bilstm: bilstmPreds
        },
        chosenWord: word,
        correctModels
      };

      setHistory(prev => [...prev, historyEvent]);
    }

    const newSentence = `${sentence} ${word}`;
    setSentence(newSentence);
    // fetchPredictions(newSentence); // Triggered by useEffect
  };

  // Text-to-Speech functionality
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voicesLoaded, setVoicesLoaded] = useState(false);

  // Ensure voices are loaded
  React.useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        setVoicesLoaded(true);
        console.log('Available TTS voices:', voices.map(v => `${v.name} (${v.lang})`));
      }
    };

    // Load voices immediately
    loadVoices();

    // Some browsers need this event
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }, []);


  const speakText = (text: string) => {
    if (!text.trim()) return;

    // Check if browser supports speech synthesis
    if (!('speechSynthesis' in window)) {
      alert('Sorry, your browser does not support text-to-speech.');
      return;
    }

    // Stop any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);

    // Set language based on current mode
    if (language === 'ta') {
      // Try to find a Tamil voice
      const voices = window.speechSynthesis.getVoices();
      const tamilVoice = voices.find(voice =>
        voice.lang.startsWith('ta') ||
        voice.lang.includes('Tamil') ||
        voice.name.includes('Tamil')
      );

      if (tamilVoice) {
        utterance.voice = tamilVoice;
        utterance.lang = tamilVoice.lang;
      } else {
        // Fallback: Try multiple Tamil language codes
        utterance.lang = 'ta-IN'; // Tamil (India)
        console.warn('No Tamil voice found. Using default with ta-IN language code.');
        console.log('Available voices:', voices.map(v => `${v.name} (${v.lang})`));
      }
    } else {
      // English
      const voices = window.speechSynthesis.getVoices();
      const englishVoice = voices.find(voice =>
        voice.lang.startsWith('en')
      );

      if (englishVoice) {
        utterance.voice = englishVoice;
        utterance.lang = englishVoice.lang;
      } else {
        utterance.lang = 'en-US';
        console.warn('No English voice found. Using default with en-US language code.');
      }
    }

    utterance.rate = 0.9; // Slightly slower for clarity
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Event handlers
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = (event) => {
      setIsSpeaking(false);
      console.error('Speech synthesis error:', event);

      if (language === 'ta') {
        alert('Tamil voice not available on this device. Please install Tamil language support in your operating system settings.\n\nWindows: Settings → Time & Language → Language → Add Tamil\nMac: System Preferences → Accessibility → Spoken Content → System Voice → Manage Voices');
      } else {
        console.warn('English speech synthesis encountered an error, but will attempt to continue with available voices.');
      }
    };

    window.speechSynthesis.speak(utterance);
  };

  const stopSpeaking = () => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  };

  // Keyboard Shortcuts
  const [showShortcutsHelp, setShowShortcutsHelp] = useState(false);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in textarea or input
      const target = e.target as HTMLElement;
      if (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT') {
        return;
      }

      // Don't trigger if no predictions available
      if (!results || !started) return;

      // Get available predictions based on language
      const predictions = language === 'ta'
        ? [
          results.bi_gru?.predictions?.[0]?.word,
          results.lstm?.predictions?.[0]?.word,
          results.bilstm?.predictions?.[0]?.word
        ]
        : [
          results.lstm?.predictions?.[0]?.word,
          results.bilstm?.predictions?.[0]?.word
        ];

      // Number keys 1-3 for quick selection
      if (e.key === '1' && predictions[0]) {
        e.preventDefault();
        handleWordClick(predictions[0]);
      } else if (e.key === '2' && predictions[1]) {
        e.preventDefault();
        handleWordClick(predictions[1]);
      } else if (e.key === '3' && predictions[2]) {
        e.preventDefault();
        handleWordClick(predictions[2]);
      }
      // Ctrl+Space for TTS
      else if (e.ctrlKey && e.code === 'Space') {
        e.preventDefault();
        if (isSpeaking) {
          stopSpeaking();
        } else {
          speakText(sentence);
        }
      }
      // Escape to stop TTS
      else if (e.key === 'Escape' && isSpeaking) {
        e.preventDefault();
        stopSpeaking();
      }
      // ? to show shortcuts help
      else if (e.key === '?' && !e.shiftKey) {
        e.preventDefault();
        setShowShortcutsHelp(!showShortcutsHelp);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [results, started, language, sentence, isSpeaking, showShortcutsHelp]);

  // Export Conversation
  const exportConversation = () => {
    if (!sentence.trim() && history.length === 0) {
      alert('No conversation to export. Start building a sentence first!');
      return;
    }

    // Build export content
    let content = '=== AAC Prediction Session Export ===\n\n';
    content += `Date: ${new Date().toLocaleString()}\n`;
    content += `Language: ${language === 'ta' ? 'Tamil' : 'English'}\n`;
    content += `Prefix Mode: ${prefixMode ? 'Enabled' : 'Disabled'}\n\n`;

    content += '--- CURRENT SENTENCE ---\n';
    content += sentence || '(empty)';
    content += '\n\n';

    if (history.length > 0) {
      content += '--- PREDICTION HISTORY ---\n\n';
      history.forEach((event, idx) => {
        content += `${idx + 1}. [${event.timestamp.toLocaleTimeString()}]\n`;
        content += `   Context: ${event.context || '(start)'}\n`;
        content += `   Chosen: ${event.chosenWord}\n`;
        content += `   Predicted by: ${event.correctModels.length > 0 ? event.correctModels.join(', ') : 'None (manual entry)'}\n`;

        if (language === 'ta') {
          content += `   Bi-GRU: ${event.predictions.bi_gru.slice(0, 3).join(', ')}\n`;
        }
        content += `   LSTM: ${event.predictions.lstm.slice(0, 3).join(', ')}\n`;
        content += `   BiLSTM: ${event.predictions.bilstm.slice(0, 3).join(', ')}\n\n`;
      });

      // Add statistics
      const totalSelections = history.length;
      const correctPredictions = history.filter(h => h.correctModels.length > 0).length;
      const accuracy = totalSelections > 0 ? ((correctPredictions / totalSelections) * 100).toFixed(1) : 0;

      content += '--- STATISTICS ---\n';
      content += `Total word selections: ${totalSelections}\n`;
      content += `Correctly predicted: ${correctPredictions}\n`;
      content += `Prediction accuracy: ${accuracy}%\n`;
    }

    content += '\n--- END OF EXPORT ---\n';

    // Create and download file
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aac-conversation-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Default empty result for initial render if null
  const defaultResult: ModelResult = { predictions: [], top1_confidence: 0, validity_score: 0 };

  return (
    <div className="min-h-screen bg-black text-gray-100 font-sans selection:bg-purple-500 selection:text-white pb-20 pt-20">
      {/* 0. Fixed Navbar */}
      <nav className="fixed top-0 left-0 right-0 bg-black/90 backdrop-blur-md border-b border-gray-800 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="font-bold text-xl tracking-tight text-white">
            AAC <span className="text-gray-500">Predict</span>
          </div>

          <div className="flex items-center gap-6">
            {/* Language Selector */}
            <div className="flex gap-1 bg-gray-900/50 p-1 rounded-lg border border-gray-800">
              <button
                onClick={() => switchLanguage('ta')}
                className={`px-6 py-2 rounded-md text-sm font-medium transition-all ${language === 'ta'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
              >
                Tamil
              </button>
              <button
                onClick={() => switchLanguage('en')}
                className={`px-6 py-2 rounded-md text-sm font-medium transition-all ${language === 'en'
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/20'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
              >
                English
              </button>
            </div>

            {/* Prefix Mode Toggle */}
            <div className="flex items-center gap-3 bg-gray-900/50 px-4 py-2 rounded-lg border border-gray-800">
              <span className="text-sm text-gray-400 font-medium">Prefix Mode</span>
              <button
                onClick={() => setPrefixMode(!prefixMode)}
                className={`relative w-12 h-6 rounded-full transition-all duration-300 ${prefixMode ? 'bg-purple-600' : 'bg-gray-700'
                  }`}
                aria-label="Toggle prefix filtering mode"
              >
                <div
                  className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-300 ${prefixMode ? 'translate-x-6' : 'translate-x-0'
                    }`}
                />
              </button>
              <span className={`text-xs font-medium ${prefixMode ? 'text-purple-400' : 'text-gray-500'}`}>
                {prefixMode ? 'ON' : 'OFF'}
              </span>
            </div>

            {/* Keyboard Shortcuts Help Button */}
            <button
              onClick={() => setShowShortcutsHelp(!showShortcutsHelp)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 transition-all"
              title="Keyboard shortcuts"
            >
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
              <span className="text-xs text-gray-400">?</span>
            </button>
          </div>
        </div>
      </nav>

      {/* Keyboard Shortcuts Help Panel */}
      {showShortcutsHelp && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowShortcutsHelp(false)}>
          <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-md w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold text-white">Keyboard Shortcuts</h3>
              <button onClick={() => setShowShortcutsHelp(false)} className="text-gray-400 hover:text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center py-2 border-b border-gray-800">
                <span className="text-gray-400">Select 1st prediction</span>
                <kbd className="px-2 py-1 bg-gray-800 rounded text-purple-400 font-mono">1</kbd>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-800">
                <span className="text-gray-400">Select 2nd prediction</span>
                <kbd className="px-2 py-1 bg-gray-800 rounded text-purple-400 font-mono">2</kbd>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-800">
                <span className="text-gray-400">Select 3rd prediction</span>
                <kbd className="px-2 py-1 bg-gray-800 rounded text-purple-400 font-mono">3</kbd>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-800">
                <span className="text-gray-400">Speak sentence</span>
                <kbd className="px-2 py-1 bg-gray-800 rounded text-purple-400 font-mono">Ctrl + Space</kbd>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-800">
                <span className="text-gray-400">Stop speaking</span>
                <kbd className="px-2 py-1 bg-gray-800 rounded text-purple-400 font-mono">Esc</kbd>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-gray-400">Toggle this help</span>
                <kbd className="px-2 py-1 bg-gray-800 rounded text-purple-400 font-mono">?</kbd>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-4">
              💡 Shortcuts work when not typing in text fields
            </p>
          </div>
        </div>
      )}

      {/* 1. Header */}
      <header className="p-8 pb-12 bg-gradient-to-b from-gray-900 to-black border-b border-gray-800">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 mb-2">
            {language === 'ta' ? 'Tamil AAC Next-Word Prediction' : 'English Next-Word Prediction'}
          </h1>
          <p className="text-xl text-gray-400 font-light">
            {language === 'ta'
              ? 'Interactive Model Playground & Performance Comparison'
              : 'Direct LSTM & BiLSTM Prediction Demo (No Translation)'}
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 -mt-8">

        {/* 2. Input Section */}
        <div className="bg-gray-900/80 backdrop-blur-lg border border-gray-700 rounded-2xl p-6 shadow-2xl mb-8">
          <label className="block text-sm font-medium text-gray-400 mb-2">Initial Context</label>
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={started}
              placeholder={language === 'ta' ? "Ex: நான் இன்று" : "Ex: I am going"}
              className="flex-1 bg-black/50 border border-gray-600 rounded-lg px-4 py-3 text-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            {!started && (
              <button
                onClick={handleStart}
                disabled={!input.trim() || loading}
                className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(37,99,235,0.3)]"
              >
                {loading ? "Starting..." : "Start Prediction"}
              </button>
            )}
          </div>
        </div>

        {/* 3. Current Sentence Display (Editable) */}
        {started && (
          <div className="mb-12">
            <h2 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-3">Current Sentence Construction</h2>
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 min-h-[120px] flex items-center shadow-inner relative focus-within:ring-2 focus-within:ring-blue-500/50 transition-all">
              <textarea
                value={sentence}
                onChange={(e) => {
                  setSentence(e.target.value);
                  // Debounce logic could be here, but for simplicity/responsiveness in this demo:
                  // We'll trigger prediction on typical sentence pauses or let the user type.
                  // Actually, to support "dynamic predicting", let's use a timeout.
                }}
                onBlur={() => {
                  // Trigger prediction on blur (focus loss) to ensure latest state is predicted
                  if (sentence.trim()) fetchPredictions(sentence);
                }}
                className="w-full bg-transparent border-none text-2xl md:text-3xl text-white outline-none resize-none font-medium leading-relaxed font-inherit h-full min-h-[80px]"
                placeholder="Type here..."
                spellCheck={false}
              />
              {/* Optional: Manual Predict Button inside if needed, or rely on blur/debouncing. 
                  User said "dynamic value entering and predicting". 
                  Let's add a small effect for debouncing.
              */}
            </div>
            <div className="flex justify-between items-center mt-2">
              {/* TTS Speaker Button */}
              <button
                onClick={() => isSpeaking ? stopSpeaking() : speakText(sentence)}
                disabled={!sentence.trim()}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all disabled:opacity-30 disabled:cursor-not-allowed ${isSpeaking
                  ? 'bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-900/30'
                  : 'bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-900/30'
                  }`}
                title={isSpeaking ? 'Stop speaking' : 'Speak sentence aloud'}
              >
                {isSpeaking ? (
                  <>
                    <svg className="w-5 h-5 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
                      <rect x="6" y="4" width="3" height="12" />
                      <rect x="11" y="4" width="3" height="12" />
                    </svg>
                    <span className="text-sm">Stop</span>
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 3.75a2 2 0 00-2 2v8.5a2 2 0 004 0v-8.5a2 2 0 00-2-2zM3.5 9.75a.75.75 0 01.75.75v1a5.75 5.75 0 0011.5 0v-1a.75.75 0 011.5 0v1a7.25 7.25 0 01-6.5 7.208v1.292h2.25a.75.75 0 010 1.5h-5.5a.75.75 0 010-1.5H9.5v-1.292A7.25 7.25 0 013 11.5v-1a.75.75 0 01.75-.75z" />
                    </svg>
                    <span className="text-sm">Speak</span>
                  </>
                )}
              </button>

              <div className="flex gap-3">
                <button
                  onClick={exportConversation}
                  disabled={!sentence.trim() && history.length === 0}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 text-white font-medium transition-all disabled:opacity-30 disabled:cursor-not-allowed shadow-lg shadow-green-900/30"
                  title="Export conversation to text file"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  <span className="text-sm">Export</span>
                </button>

                <button
                  onClick={() => fetchPredictions(sentence)}
                  className="text-sm text-blue-400 hover:text-blue-300 font-medium cursor-pointer"
                >
                  ↻ Refresh Predictions
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 4. Model Cards Grid */}
        {started && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            {language === 'ta' && (
              <ModelCard
                title="Direct Tamil Bi-GRU"
                subtitle="Native Next-Word Prediction"
                data={results?.bi_gru || defaultResult}
                color="blue"
                onWordClick={handleWordClick}
                loading={loading}
              />
            )}

            <ModelCard
              title={language === 'ta' ? "LSTM Baseline" : "LSTM (English)"}
              subtitle={language === 'ta' ? "Translation-Assisted Pipeline" : "Baseline Model"}
              pipeline={language === 'ta' ? "Ta → En → LSTM → En → Ta" : "English Only"}
              data={results?.lstm || defaultResult}
              color="emerald"
              onWordClick={handleWordClick}
              loading={loading}
            />

            <ModelCard
              title={language === 'ta' ? "BiLSTM Model" : "BiLSTM (English)"}
              subtitle={language === 'ta' ? "Enhanced Context Baseline" : "Bidirectional Model"}
              pipeline={language === 'ta' ? "Ta → En → BiLSTM → En → Ta" : "English Only"}
              data={results?.bilstm || defaultResult}
              color="amber"
              onWordClick={handleWordClick}
              loading={loading}
            />
          </div>
        )}

        {/* 4.5 Prediction History Timeline */}
        {started && history.length > 0 && (
          <div className="mb-12">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-bold text-gray-500 uppercase tracking-widest">Prediction History</h2>
              <button
                onClick={() => setHistory([])}
                className="text-xs text-red-400 hover:text-red-300 font-medium"
              >
                Clear History
              </button>
            </div>
            <div className="bg-gray-900/80 backdrop-blur-lg border border-gray-700 rounded-2xl p-6 max-h-96 overflow-y-auto">
              <div className="space-y-4">
                {history.map((event, idx) => (
                  <div key={idx} className="border-l-4 border-purple-500 pl-4 py-2 bg-gray-800/50 rounded-r-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <span className="text-xs text-gray-500">
                          {event.timestamp.toLocaleTimeString()}
                        </span>
                        <p className="text-sm text-gray-400 mt-1">
                          Context: <span className="text-white">{event.context || '(start)'}</span>
                        </p>
                      </div>
                      <div className="text-right">
                        <span className="text-lg font-bold text-purple-400">{event.chosenWord}</span>
                        <p className="text-xs text-gray-500 mt-1">Selected</p>
                      </div>
                    </div>

                    {/* Model Predictions */}
                    <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
                      {language === 'ta' && (
                        <div className={`p-2 rounded ${event.correctModels.includes('Bi-GRU') ? 'bg-green-900/30 border border-green-700' : 'bg-gray-700/30'}`}>
                          <p className="font-medium text-blue-400 mb-1">Bi-GRU</p>
                          <p className="text-gray-300">{event.predictions.bi_gru.slice(0, 3).join(', ') || 'N/A'}</p>
                        </div>
                      )}
                      <div className={`p-2 rounded ${event.correctModels.includes('LSTM') ? 'bg-green-900/30 border border-green-700' : 'bg-gray-700/30'}`}>
                        <p className="font-medium text-emerald-400 mb-1">LSTM</p>
                        <p className="text-gray-300">{event.predictions.lstm.slice(0, 3).join(', ') || 'N/A'}</p>
                      </div>
                      <div className={`p-2 rounded ${event.correctModels.includes('BiLSTM') ? 'bg-green-900/30 border border-green-700' : 'bg-gray-700/30'}`}>
                        <p className="font-medium text-amber-400 mb-1">BiLSTM</p>
                        <p className="text-gray-300">{event.predictions.bilstm.slice(0, 3).join(', ') || 'N/A'}</p>
                      </div>
                    </div>

                    {/* Accuracy Indicator */}
                    {event.correctModels.length > 0 ? (
                      <p className="text-xs text-green-400 mt-2">
                        ✓ Predicted by: {event.correctModels.join(', ')}
                      </p>
                    ) : (
                      <p className="text-xs text-orange-400 mt-2">
                        ⚠ Not in top predictions (user typed manually or different choice)
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 5. Visualization Section */}
        {started && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
            <div className="lg:col-span-2 space-y-8">
              <ConfidenceGraph stats={stats} />
              <ValidityGraph stats={stats} />
            </div>

            {/* 6. Static Stats */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-200 mb-4">Dataset & Model Info</h3>
              <div className="space-y-4 text-sm text-gray-400">
                <div className="flex justify-between border-b border-gray-800 pb-2">
                  <span>Language Mode</span>
                  <span className="text-white capitalize">{language === 'ta' ? 'Tamil' : 'English'}</span>
                </div>
                <div className="flex justify-between border-b border-gray-800 pb-2">
                  <span>Vocab Size</span>
                  <span className="text-white">{language === 'ta' ? '1100' : '1500'}</span>
                </div>
                <div className="flex justify-between border-b border-gray-800 pb-2">
                  <span>Training Set</span>
                  <span className="text-white">{language === 'ta' ? '50K Sentences' : '700K Sentences'}</span>
                </div>
                <div className="flex justify-between border-b border-gray-800 pb-2">
                  <span>{language === 'ta' ? 'Bi-GRU Architecture' : 'LSTM / BiLSTM Source'}</span>
                  <span className="text-white">{language === 'ta' ? 'Custom Native' : 'English Common Crawl'}</span>
                </div>

                {language === 'ta' ? (
                  <div className="pt-2">
                    <p className="mb-2">Bi-GRU Details:</p>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">Embedding (300d)</span>
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">Bi-GRU (256)</span>
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">Dense (Vocab)</span>
                    </div>
                  </div>
                ) : (
                  <div className="pt-2">
                    <p className="mb-2">Model Details:</p>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">LSTM (Baseline)</span>
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">BiLSTM (Context)</span>
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">Glove Embeddings</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="text-center text-gray-600 text-sm mt-20">
          <p>Note: Accuracy values represent Top-3 accuracy. Single-word comparison metrics may vary based on translation ambiguity.</p>
        </footer>

      </main>
    </div>
  );
}

export default App;
