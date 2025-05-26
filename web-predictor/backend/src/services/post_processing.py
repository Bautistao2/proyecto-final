from typing import Dict, Tuple, List
import numpy as np

def post_process_prediction(prediction: Dict) -> Dict:
    """
    Post-processes model predictions to ensure the wing is adjacent to the type according to Enneagram theory.
    
    Args:
        prediction: A dictionary containing the model's raw predictions with keys:
                   'enneagram_type', 'wing', 'type_probabilities', 'wing_probabilities'
    
    Returns:
        A dictionary with the corrected prediction
    """
    enneagram_type = prediction['enneagram_type']
    wing_probs = prediction['wing_probabilities']
    
    # Determine valid adjacent wings for the predicted type
    valid_wings = get_valid_wings(enneagram_type)
    
    # If the current wing is valid, keep it
    if prediction['wing'] in valid_wings:
        return prediction
    
    # Otherwise, select the valid wing with highest probability
    valid_wing_probs = {wing: wing_probs[wing-1] for wing in valid_wings}
    new_wing = max(valid_wing_probs, key=valid_wing_probs.get)
    
    # Return corrected prediction
    corrected = prediction.copy()
    corrected['wing'] = new_wing
    return corrected

def get_valid_wings(enneagram_type: int) -> List[int]:
    """
    Returns valid wings for a given Enneagram type.
    Wings must be adjacent to the type in the Enneagram circle.
    
    Args:
        enneagram_type: An integer from 1 to 9 representing the Enneagram type
    
    Returns:
        A list of valid wing numbers (either one or two numbers)
    """
    if enneagram_type < 1 or enneagram_type > 9:
        raise ValueError("Enneagram type must be between 1 and 9")
    
    # For type 9, the valid wings are 8 and 1 (circle wraps around)
    if enneagram_type == 9:
        return [8, 1]
    # For type 1, the valid wings are 9 and 2
    elif enneagram_type == 1:
        return [9, 2]
    # For all other types, wings are the numbers before and after
    else:
        return [enneagram_type - 1, enneagram_type + 1]
