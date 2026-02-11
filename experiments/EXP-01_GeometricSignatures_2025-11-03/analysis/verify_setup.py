import google.generativeai as genai
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import os

try:
    # Setup API Key (using the same one for verification)
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
    genai.configure(api_key=GOOGLE_API_KEY)

    print("Dependencies imported.")

    text = "Verification test."
    print(f"Getting embedding for: {text}")
    model = "models/text-embedding-004"
    result = genai.embed_content(model=model, content=text)
    embedding = np.array(result['embedding'])
    
    print(f"Embedding shape: {embedding.shape}")

    # Test distances
    dist = cosine(embedding, embedding)
    print(f"Self cosine distance: {dist}")
    
    if dist < 1e-6:
        print("VERIFICATION SUCCESS: API and Distance functions working.")
    else:
        print("VERIFICATION FAILED: Distance check failed.")

except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
