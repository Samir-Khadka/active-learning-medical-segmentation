import sys
import os

# Ensure the project root is in the path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Ensure the src directory is in the path for internal imports
sys.path.insert(0, os.path.join(root_dir, 'src'))

from src.main import main

if __name__ == "__main__":
    main()
