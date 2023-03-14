MathJax.Hub.Config({
    TeX: {
        Macros:
        {
            bs: ["{\\boldsymbol #1}", 1],
            d: ["{\\mathrm{d}#1}",1],
            dint: ["{\\,\\mathrm{d}#1}", 1],
            subindex: [" #1\_\{\\mathrm \{ #2 \} \}", 2],

            gpi: ["\\text\{ \\\greektext p\}\}"],
            gtheta: ["\\text \{ \\\greektext j \}"],
            gmu: ["\\text \{ \\greektext m \}"],
            geta: ["\\text \{ \\greektext h \}"],
            gLambda: ["\\text \{ \\greektext L\}"],
            laplace: ["\\text \{ \\greektext D \}"],

            ex: ["\\boldsymbol \{ e \}\_x"],
            ey: ["\\boldsymbol \{ e \}\_y"],
            ez: ["\\boldsymbol \{ e \}\_z"],

            etheta: ["\\boldsymbol \{ e \}\_\\theta"],
            ephi: ["\\boldsymbol \{ e \}\_\\varphi"],

            p: ["\\partial"],
            dd: ["\\frac{\\mathrm{d} #1 \} \{ \\mathrm{d} #2 \}", 2],
            ddsqr: ["\\frac{\\mathrm{d}^2 #1\} \{ \\mathrm{d} #2^2 \}", 2],
            pd: ["\\frac{\\p #1 \} \{\\p #2\}", 2],
            ppd: ["\\frac{\\p^2 #1\} \{\\p #2 \\p #3 \}", 3],
            pdsqr: ["\\frac{\\p^2 #1\} \{ \\p #2^2 \}", 2],

            cdott: ["\\, \{\\cdot\} \{ \\cdot \}\\,"],
            abs: ["\\left\\lvert #1 \\right\\rvert", 1],
            sgn: ["\\operatorname\{sgn\}"],
            erf: ["\\operatorname\{erf\}"],
            gd: ["\\operatorname\{gd\}"],
            atan: ["\\operatorname\{atan\}"],
            vol: ["\\operatorname\{vol\}"],
            macaulay: ["\\left\\langle #1 \\right\\rangle", 1],
            ts : ["\\overset{\\hspace{1pt}\\scriptscriptstyle\\langle#2\\rangle}{\\boldsymbol{#1}}", 2],
            overbar:["\\mkern 1.5mu\\overline{\\mkern-1.5mu#1\\mkern-1.5mu}\\mkern 1.5mu", 1],

        }
    }
});

