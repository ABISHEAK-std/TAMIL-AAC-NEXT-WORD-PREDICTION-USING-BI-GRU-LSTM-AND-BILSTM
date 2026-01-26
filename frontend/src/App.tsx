import React, { useState } from 'react';
import axios from 'axios';
import ModelCard from './components/ModelCard';
import ConfidenceGraph from './components/ConfidenceGraph';
import ValidityGraph from './components/ValidityGraph';
import type { PredictionResponse, GlobalStats, ModelResult } from './types';

// API CONSTANTS
const API_URL = 'http://localhost:8000';

function App() {
  /* New State for Language */
  const [language, setLanguage] = useState<'ta' | 'en'>('ta');

  const [input, setInput] = useState('');
  const [sentence, setSentence] = useState('');
  const [started, setStarted] = useState(false);
  /* Debounced Auto-Prediction Logic */
  const [loading, setLoading] = useState(false);
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
    try {
      const res = await axios.post(`${API_URL}/predict`, {
        current_sentence: currentText,
        language: language
      });
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

    } catch (err) {
      console.error("Prediction Error:", err);
      alert("Failed to fetch predictions. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleStart = () => {
    if (!input.trim()) return;
    setSentence(input);
    setStarted(true);
    // fetchPredictions(input); // Triggered by useEffect
  };

  const handleWordClick = (word: string) => {
    const newSentence = `${sentence} ${word}`;
    setSentence(newSentence);
    // fetchPredictions(newSentence); // Triggered by useEffect
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
        </div>
      </nav>

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
            <div className="flex justify-end mt-2">
              <button
                onClick={() => fetchPredictions(sentence)}
                className="text-sm text-blue-400 hover:text-blue-300 font-medium cursor-pointer"
              >
                ↻ Refresh Predictions
              </button>
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
