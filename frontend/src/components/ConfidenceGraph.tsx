import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { GlobalStats } from '../types';

interface ConfidenceGraphProps {
    stats: GlobalStats;
}

const ConfidenceGraph: React.FC<ConfidenceGraphProps> = ({ stats }) => {
    // Transform data for Recharts
    // stats.bi_gru_conf is [val1, val2...]
    // We need [{ step: 1, bigru: val1, lstm: val1, bilstm: val1 }, ...]

    const maxLength = Math.max(
        stats.bi_gru_conf.length,
        stats.lstm_conf.length,
        stats.bilstm_conf.length
    );

    const data = Array.from({ length: maxLength }).map((_, i) => ({
        step: i + 1,
        'Direct Tamil Bi-GRU': isNaN(stats.bi_gru_conf[i]) ? 0 : (stats.bi_gru_conf[i] || 0) * 100,
        'LSTM Baseline': isNaN(stats.lstm_conf[i]) ? 0 : (stats.lstm_conf[i] || 0) * 100,
        'BiLSTM Model': isNaN(stats.bilstm_conf[i]) ? 0 : (stats.bilstm_conf[i] || 0) * 100,
    }));

    if (maxLength === 0) {
        return (
            <div className="flex items-center justify-center h-[300px] bg-gray-900/50 rounded-xl border border-gray-800">
                <p className="text-gray-500">Start prediction to see confidence growth</p>
            </div>
        );
    }

    return (
        <div className="w-full h-[350px] p-4 bg-gray-900 rounded-xl border border-gray-800 shadow-xl">
            <h3 className="text-lg font-bold text-gray-200 mb-4">Model Confidence Over Prediction Steps</h3>
            <ResponsiveContainer width="100%" height="90%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="step" stroke="#9CA3AF" label={{ value: 'Step', position: 'insideBottomRight', offset: -5 }} />
                    <YAxis stroke="#9CA3AF" domain={[0, 100]} label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#F3F4F6' }}
                        itemStyle={{ color: '#F3F4F6' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="Direct Tamil Bi-GRU" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                    <Line type="monotone" dataKey="LSTM Baseline" stroke="#10B981" strokeWidth={3} dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="BiLSTM Model" stroke="#F59E0B" strokeWidth={3} dot={{ r: 4 }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ConfidenceGraph;
