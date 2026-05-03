import sys
import os

# Add the project root and src to the path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

import uvicorn

if __name__ == "__main__":
    print("[INFO] Starting Professional Fullstack Active Learning Platform...")
    print("[INFO] Dashboard available at: http://localhost:8000")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
