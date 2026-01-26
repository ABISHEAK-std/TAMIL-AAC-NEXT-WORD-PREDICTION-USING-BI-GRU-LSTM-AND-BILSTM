export interface Prediction {
    word: string;
    confidence: number;
}

export interface ModelResult {
    predictions: Prediction[];
    top1_confidence: number;
    validity_score: number;
}

export interface PredictionResponse {
    bi_gru: ModelResult;
    lstm: ModelResult;
    bilstm: ModelResult;
}

export interface GlobalStats {
    bi_gru_conf: number[];
    lstm_conf: number[];
    bilstm_conf: number[];
    bi_gru_valid: number[];
    lstm_valid: number[];
    bilstm_valid: number[];
    labels: string[];
}
