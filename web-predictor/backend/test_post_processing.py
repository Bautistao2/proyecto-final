import sys
from pathlib import Path
import numpy as np
import random

# Configure paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))  # Add backend directory to path
sys.path.insert(0, str(BASE_DIR / 'src'))  # Add src directory to path

# Import the post_processing module
from src.services.post_processing import post_process_prediction, get_valid_wings

def test_post_processing():
    """
    Test the post-processing function to ensure that wings are always adjacent to types.
    """
    print("=== Testing Post-Processing for Enneagram Predictions ===")
    
    # Test all type-wing combinations
    results = []
    
    for enneagram_type in range(1, 10):
        valid_wings = get_valid_wings(enneagram_type)
        print(f"\nTesting Enneagram Type {enneagram_type}")
        print(f"  Valid wings: {valid_wings}")
        
        # Test each valid wing (should remain unchanged)
        for wing in valid_wings:
            # Create a mock prediction
            prediction = {
                "enneagram_type": enneagram_type,
                "wing": wing,
                "type_probabilities": [random.random() for _ in range(9)],
                "wing_probabilities": [random.random() for _ in range(9)]
            }
            
            # Apply post-processing
            result = post_process_prediction(prediction)
            
            # Check if wing is unchanged (should be)
            if result["wing"] == wing:
                print(f"  ✅ Valid combination {enneagram_type}w{wing}: Passed")
                results.append(True)
            else:
                print(f"  ❌ Valid combination {enneagram_type}w{wing}: Failed - changed to {result['wing']}")
                results.append(False)
        
        # Test with an invalid wing
        invalid_wings = [w for w in range(1, 10) if w not in valid_wings]
        for wing in invalid_wings:
            # Set highest probability for this invalid wing
            wing_probs = [0.1] * 9
            wing_probs[wing - 1] = 0.9
            
            # Create mock prediction with invalid wing
            prediction = {
                "enneagram_type": enneagram_type,
                "wing": wing,
                "type_probabilities": [random.random() for _ in range(9)],
                "wing_probabilities": wing_probs
            }
            
            # Apply post-processing
            result = post_process_prediction(prediction)
            
            # Check if wing is corrected
            if result["wing"] in valid_wings:
                print(f"  ✅ Invalid combination {enneagram_type}w{wing}: Corrected to {result['wing']}")
                results.append(True)
            else:
                print(f"  ❌ Invalid combination {enneagram_type}w{wing}: Failed - still invalid: {result['wing']}")
                results.append(False)
    
    # Print overall results
    success_rate = sum(results) / len(results) * 100
    print(f"\nTest Results: {sum(results)}/{len(results)} tests passed ({success_rate:.1f}%)")
    
    return all(results)

if __name__ == "__main__":
    success = test_post_processing()
    print(f"\nTest {'succeeded' if success else 'failed'}")
