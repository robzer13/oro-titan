import os
import sys

# pour être sûr que /oro_titan est dans le path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "indicators"))

from indicators import build_all_indicators

def main():
    df = build_all_indicators("AAPL", start="2018-01-01", benchmark="^GSPC")
    print(df.tail(10))

if __name__ == "__main__":
    main()
