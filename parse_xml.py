from pitchplots.parser import xml_to_csv
import pandas as pd
import sys

if __name__ == "__main__":
    for f in sys.argv[1:]:
        name, ext = f.rsplit(".", 1)
        df = xml_to_csv(f)
        df.to_csv(name+".csv")
