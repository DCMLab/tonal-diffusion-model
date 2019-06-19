from pitchplots.parser import xml_to_csv
import pandas as pd
import sys

if __name__ == "__main__":
    infile = sys.argv[1]
    name, ext = infile.split(".")
    df = xml_to_csv(infile)
    df.to_csv(name+".csv")
