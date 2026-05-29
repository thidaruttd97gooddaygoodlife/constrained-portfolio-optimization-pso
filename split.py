import os
import re

def main():
    with open('portfolio_nsga2.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # We will just write the files directly here to ensure accuracy.
    # Actually, writing a regex to split is risky if the comments aren't exact.
    # Let's extract specific classes using ast or just string manipulation.
    pass

if __name__ == "__main__":
    main()
