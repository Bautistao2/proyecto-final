"""
Test script for the Enneagram Prediction API

This script is used to test both the /questions and /predict endpoints of the API.
"""
import requests
import json
import sys
import numpy as np
from typing import Dict, List, Any

BASE_URL = "http://localhost:8000/api/v1"

def test_questions_endpoint():
    """Test the /questions endpoint"""
    print("\n=== Testing Questions Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/questions")
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Total questions: {data.get('total', 0)}")
        print(f"First question: {data['questions'][0] if data.get('questions') else 'No questions'}")
        
        if data.get('questions'):
            return True, data['questions']
        else:
            print("No questions returned")
            return False, []
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, []

def test_predict_endpoint(answers=None):
    """Test the /predict endpoint"""
    print("\n=== Testing Predict Endpoint ===")
    
    if not answers:
        # Create default answers (all neutral)
        answers = [3] * 80
    
    try:
        print(f"Sending {len(answers)} answers to prediction endpoint...")
        response = requests.post(
            f"{BASE_URL}/predict", 
            json={"answers": answers}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            return True, result
        else:
            print(f"Error response: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, None

def run_tests():
    """Run all API tests"""
    print("=== Starting API Tests ===")
    
    questions_success, questions = test_questions_endpoint()
    predict_success, prediction = test_predict_endpoint()
    
    print("\n=== Test Results ===")
    print(f"Questions endpoint: {'✅ PASS' if questions_success else '❌ FAIL'}")
    print(f"Predict endpoint: {'✅ PASS' if predict_success else '❌ FAIL'}")
    
    if not (questions_success and predict_success):
        print("\nSome tests failed. Please check the backend server.")
        return False
    
    print("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
