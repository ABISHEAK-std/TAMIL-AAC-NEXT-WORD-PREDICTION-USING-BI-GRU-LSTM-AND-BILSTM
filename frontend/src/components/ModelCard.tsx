import React from 'react';
import { motion } from 'framer-motion';
import type { ModelResult } from '../types';

interface ModelCardProps {
    title: string;
    subtitle: string;
    pipeline?: string; // Optional pipeline description for cards 2 & 3
    data: ModelResult;
    color: string; // Tailwind border/accent color class
    onWordClick: (word: string) => void;
    loading: boolean;
}

const ModelCard: React.FC<ModelCardProps> = ({
    title, subtitle, pipeline, data, color, onWordClick, loading
}) => {
    return (
        <div className={`flex flex-col p-6 rounded-xl bg-gray-900 border border-gray-800 shadow-xl min-h-[300px] relative overflow-hidden group`}>
            {/* Decorative Gradient Blob */}
            <div className={`absolute top-0 right-0 w-32 h-32 bg-${color}-500 opacity-5 blur-[60px] rounded-full group-hover:opacity-10 transition-opacity duration-500`} />

            <div className="z-10 flex flex-col h-full">
                <h3 className={`text-xl font-bold text-${color}-400 mb-1`}>{title}</h3>
                <p className="text-xs text-gray-400 uppercase tracking-wider mb-4">{subtitle}</p>

                {pipeline && (
                    <div className="mb-4 p-2 bg-black/40 rounded border border-gray-800 text-xs text-gray-300 font-mono">
                        {pipeline}
                    </div>
                )}

                <div className="flex-1 flex flex-col gap-3 mt-2">
                    <p className="text-sm text-gray-400 mb-1">Top 3 Predictions:</p>

                    {loading ? (
                        <div className="space-y-3 animate-pulse">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="h-10 bg-gray-800 rounded w-full"></div>
                            ))}
                        </div>
                    ) : (
                        data.predictions.map((pred, idx) => (
                            <motion.button
                                key={`${pred.word}-${idx}`}
                                whileHover={{ scale: 1.02, x: 5 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => onWordClick(pred.word)}
                                className={`flex items-center justify-between w-full p-3 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 transition-all text-left group-hover:border-${color}-900`}
                            >
                                <span className="text-lg font-medium text-white">{pred.word}</span>
                                <span className={`text-xs font-mono text-${color}-400 bg-${color}-900/20 px-2 py-1 rounded`}>
                                    {(pred.confidence * 100).toFixed(1)}%
                                </span>
                            </motion.button>
                        ))
                    )}
                </div>

                {/* Footer Stats */}
                <div className="mt-6 pt-4 border-t border-gray-800 flex justify-between text-xs text-gray-500">
                    <span>Top-1 Confidence: <span className="text-white">{(data.top1_confidence * 100).toFixed(1)}%</span></span>
                    <span>Inference: ~50ms</span>
                </div>
            </div>
        </div>
    );
};

export default ModelCard;
