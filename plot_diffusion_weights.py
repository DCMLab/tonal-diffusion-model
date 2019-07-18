import pandas as pd
import sys

if __name__=="__main__":
    """Pass filename(s) as arguments. Returns tikz plots
    of parameters"""

    df = pd.read_csv("results_1.tsv", sep="\t")

    for f in sys.argv[1:]:
        piece = df[df["8"].str.split("\\").str[-1] ==f]

        # read parameter columns
        weights = piece.iloc[:,1:7].iloc[0]
        # decays = piece.iloc[:,7:13].iloc[0]
        decay = piece.iloc[:,7].iloc[0]

        intervals = ["+P5", "-P5", "+m3", "-m3", "+M3", "-M3", ]
        weights.index = intervals

        labels = ["+P5", "+m3", "-M3", "-P5", "-m3", "+M3", ]
        weights = weights.reindex(labels)
        # decays = decays.reindex(labels)

        tikzpicture = f"""
        \\documentclass[tikz]{{standalone}}
        \\begin{{document}}
        \\begin{{tikzpicture}}[scale=3.5, ->, >=stealth]

        % the circle
        \\node (origin) at (0,0) [draw,circle,black,fill=white,thick]{{}};
        % constant scaling for widths
        \\def\\cons{{10}}
        """

        eps = 10e-4
        for i in range(6):
            w = weights.iloc[i]
            # l = decays.iloc[i]

            if (w > eps) & (decay > eps):
                tikzpicture += f"""
                \\node ({i}) at +({-i}*360/6:{decay}) {{{labels[i]}}};
                \\path (origin) edge [line width=\\cons*{w}] node {{}} ({i});
                """

        tikzpicture += """
        \\end{tikzpicture}
        \\end{document}
        """

        fname = f.split(".")[0]
        with open(f"img/tikz/tikz_{fname}.tex", "w") as file:
            file.write(tikzpicture)
