import axios, { AxiosInstance } from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

// Create axios instance with default config
const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface PredictionRequest {
  answers: number[];
}

export interface PredictionResponse {
  enneagram_type: number;
  wing: number;
  type_probabilities: number[];
  wing_probabilities: number[];
}

export interface Question {
  id: number;
  text: string;
  category: string;
  original_id: string;
}

export interface QuestionsResponse {
  questions: Question[];
  total: number;
}

// API methods
export const apiService = {
  // Get all questions
  getQuestions: async (): Promise<Question[]> => {
    const response = await api.get<QuestionsResponse>('/questions');
    return response.data.questions;
  },

  // Submit test answers and get prediction
  submitAnswers: async (answers: number[]): Promise<PredictionResponse> => {
    const response = await api.post<PredictionResponse>('/predict', { answers });
    return response.data;
  },
};

export default apiService;
