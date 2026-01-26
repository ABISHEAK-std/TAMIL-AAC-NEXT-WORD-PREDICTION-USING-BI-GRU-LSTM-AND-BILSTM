import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { GlobalStats } from '../types';

interface ValidityGraphProps {
    stats: GlobalStats;
}

const ValidityGraph: React.FC<ValidityGraphProps> = ({ stats }) => {
    const maxLength = Math.max(
        stats.bi_gru_valid.length,
        stats.lstm_valid.length,
        stats.bilstm_valid.length
    );

    const data = Array.from({ length: maxLength }).map((_, i) => ({
        step: i + 1,
        'Direct Tamil Bi-GRU': stats.bi_gru_valid[i] || 0,
        'LSTM Baseline': stats.lstm_valid[i] || 0,
        'BiLSTM Model': stats.bilstm_valid[i] || 0,
    }));

    if (maxLength === 0) {
        return (
            <div className="flex items-center justify-center h-[300px] bg-gray-900/50 rounded-xl border border-gray-800">
                <p className="text-gray-500">Validity data will appear here</p>
            </div>
        );
    }

    return (
        <div className="w-full h-[350px] p-4 bg-gray-900 rounded-xl border border-gray-800 shadow-xl mt-8">
            <div className="mb-4">
                <h3 className="text-lg font-bold text-gray-200">Grammatical Validity Across Prediction Steps</h3>
                <p className="text-xs text-gray-500 mt-1">
                    *Reflects grammatical plausibility based on rule-based validation. Independent of model confidence.
                </p>
            </div>
            <ResponsiveContainer width="100%" height="85%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="step" stroke="#9CA3AF" label={{ value: 'Step', position: 'insideBottomRight', offset: -5 }} />
                    <YAxis stroke="#9CA3AF" domain={[0, 100]} label={{ value: 'Validity Rate (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#F3F4F6' }}
                        itemStyle={{ color: '#F3F4F6' }}
                    />
                    <Legend />
                    <Line type="step" dataKey="Direct Tamil Bi-GRU" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                    <Line type="step" dataKey="LSTM Baseline" stroke="#10B981" strokeWidth={3} dot={{ r: 4 }} />
                    <Line type="step" dataKey="BiLSTM Model" stroke="#F59E0B" strokeWidth={3} dot={{ r: 4 }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ValidityGraph;
