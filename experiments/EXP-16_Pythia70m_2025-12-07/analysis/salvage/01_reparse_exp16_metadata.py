import pandas as pd
import re
import os

def robust_parse(text):
    """
    Finds the first occurrence of common stopping markers and identifies the boundary.
    Returns (clean_text, boundary_index)
    """
    # Look for "Question:", "QUESTION:", "Q:", "\nQ:", or the second occurrence of "The answer is" if it repeats
    markers = [
        r"(?i)Question:", 
        r"(?i)Q:", 
        r"\nCalculate",
        r"\nCalculate",
        r"\n---"
    ]
    
    first_marker_pos = len(text)
    
    for marker in markers:
        match = re.search(marker, text)
        if match:
            if match.start() < first_marker_pos:
                first_marker_pos = match.start()
                
    # Special case: check if there's an "A: [answer]" followed by more junk
    a_match = re.search(r"A: \d+", text)
    if a_match:
        # If there is junk after the answer line
        potential_end = a_match.end()
        # But only if it's before the first marker
        if potential_end < first_marker_pos:
            # Check if there is significant text after this answer line
            after_text = text[potential_end:].strip()
            if len(after_text) > 20: 
                # If there's more than 20 chars after "A: [answer]", truncate there unless a marker found earlier
                first_marker_pos = min(first_marker_pos, potential_end)

    return text[:first_marker_pos], first_marker_pos

def main():
    metadata_path = "experiments/Experiment 16/data/metadata.csv"
    output_path = "experiments/Experiment 16/data/metadata_reparsed.csv"
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows.")
    
    reparsed_data = []
    truncations_found = 0
    
    for _, row in df.iterrows():
        original_response = str(row['response'])
        clean_text, boundary = robust_parse(original_response)
        
        if boundary < len(original_response):
            truncations_found += 1
            
        # Re-parse answer from clean text
        nums = re.findall(r'-?\d+', clean_text)
        new_parsed = int(nums[-1]) if nums else None
        new_correct = (new_parsed == row['truth']) if new_parsed is not None else False
        
        row_dict = row.to_dict()
        row_dict['response_clean'] = clean_text
        row_dict['truncation_boundary_char'] = boundary
        row_dict['parsed_new'] = new_parsed
        row_dict['correct_new'] = new_correct
        reparsed_data.append(row_dict)
        
    out_df = pd.DataFrame(reparsed_data)
    out_df.to_csv(output_path, index=False)
    
    print(f"Reparsed metadata saved to {output_path}")
    print(f"Truncations identified: {truncations_found} / {len(df)}")
    print(f"New Accuracy (CoT): {out_df[out_df['condition'] == 'cot']['correct_new'].mean():.2%}")
    print(f"New Accuracy (Direct): {out_df[out_df['condition'] == 'direct']['correct_new'].mean():.2%}")

if __name__ == "__main__":
    main()
