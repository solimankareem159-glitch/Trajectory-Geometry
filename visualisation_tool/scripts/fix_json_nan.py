"""
Fix JSON NaN
============
Reads trajectory_data.json as raw text and replaces ': NaN' with ': null'.
This bypasses JSON parsing errors.
"""

import os

FILE = r"visualisation_tool/data/trajectory_data.json"

def main():
    print(f"Reading {FILE}...")
    with open(FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Size: {len(content)} bytes")
    print("Replacing ': NaN' with ': null'...")
    
    # Replace cases
    new_content = content.replace(': NaN,', ': null,')
    new_content = new_content.replace(': NaN}', ': null}')
    new_content = new_content.replace(': Infinity,', ': null,')
    new_content = new_content.replace(': -Infinity,', ': null,')
    
    # Check for residues
    if 'NaN' in new_content:
        print("Warning: 'NaN' still found in content (might be in strings?)")
        snippet = new_content[new_content.find('NaN')-20:new_content.find('NaN')+20]
        print(f"Context: {snippet}")
    
    with open(FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print("Saved patched file.")

if __name__ == "__main__":
    main()
