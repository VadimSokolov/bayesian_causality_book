import sys
pages=int(sys.argv[1]); per=0.002252
TW,TH=7.0,10.0   # trim 7 x 10 in (the standing KDP trim); the front motif was drawn on 6 x 9 and is centred on the larger panel
s=pages*per; W=0.125+TW+s+TW+0.125; H=TH+0.25
B0=0.125; S0=0.125+TW; F0=S0+s; FE=F0+TW; OX=(TW-6)/2; OY=(TH-9)/2
FX=F0+OX; S1=6.35+OY; S2=2.2+OY; BW=TW-1.3; BX=0.65; BY=H-0.6; BC1=TW-0.35; BC0=BC1-2.0
tex=r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{mathpazo}
\usepackage{xcolor,tikz}
\usetikzlibrary{calc}
\usepackage[paperwidth=@W@in,paperheight=@H@in,margin=0in]{geometry}
\pdfpagewidth=@W@in \pdfpageheight=@H@in
\definecolor{navy}{HTML}{A3172C}
\definecolor{navy2}{HTML}{1F3150}
\definecolor{gold}{HTML}{FFFFFF}
\definecolor{pale}{HTML}{FFFFFF}
\definecolor{teal}{HTML}{F3D9DC}
\pagestyle{empty}
\setlength{\parindent}{0pt}\hyphenpenalty=10000 \exhyphenpenalty=10000
\begin{document}
\noindent\begin{tikzpicture}[x=1in,y=1in,remember picture,overlay]
\begin{scope}[shift={(current page.south west)}]
% background
\fill[navy] (0,0) rectangle (@W@,@H@);

% front motif: posterior densities of an effect, one curve per accounting
\begin{scope}[shift={(@FX@,@OY@)}]
  \foreach \m/\sd/\c/\o in {2.2/0.42/teal/0.6, 2.9/0.36/teal/0.5, 3.5/0.30/gold/1.0, 3.95/0.40/teal/0.45}{
    \fill[\c,opacity=0.18*\o] plot[domain=0.3:5.7,samples=160,smooth] (\x,{1.25+1.55*exp(-((\x-\m)^2)/(2*\sd^2))}) -- (5.7,1.25) -- (0.3,1.25) -- cycle;
    \draw[\c,line width=1.1pt,opacity=\o] plot[domain=0.3:5.7,samples=160,smooth] (\x,{1.25+1.55*exp(-((\x-\m)^2)/(2*\sd^2))});
  }
  \draw[pale,line width=0.6pt,opacity=0.6] (0.3,1.25) -- (5.7,1.25);
  % causal graph
  \begin{scope}[shift={(3.0,4.05)}]
    \tikzset{nd/.style={circle,draw=pale,line width=0.9pt,minimum size=0.42in,inner sep=0pt,text=pale,font=\large\itshape}}
    \node[nd] (X) at (0,0.75) {X};
    \node[nd] (T) at (-1.05,-0.35) {T};
    \node[nd] (Y) at (1.05,-0.35) {Y};
    \draw[gold,line width=1.3pt,->,>=latex] (T) -- (Y);
    \draw[pale,line width=0.9pt,->,>=latex] (X) -- (T);
    \draw[pale,line width=0.9pt,->,>=latex] (X) -- (Y);
  \end{scope}
  \draw[gold,line width=1.0pt] (0.6,7.2) -- (5.4,7.2);
  \node[anchor=north,text=pale,font=\fontsize{36}{40}\selectfont] at (3.0,8.65) {Bayesian};
  \node[anchor=north,text=pale,font=\fontsize{36}{40}\selectfont] at (3.0,8.0) {Causality};
  \node[anchor=north,text=gold,align=center,font=\fontsize{14}{18}\selectfont\itshape] at (3.0,7.0) {Models, Accounting and the Bayesian\\Alternative to Double Machine Learning};
  \node[anchor=south,text=pale,font=\fontsize{20}{24}\selectfont] at (3.0,0.45) {Nick Polson \quad\textcolor{gold}{\textbullet}\quad Vadim Sokolov};
\end{scope}
% spine
\node[rotate=-90,text=pale,font=\fontsize{20}{24}\selectfont] at ($(@S0@,0)!0.5!(@F0@,0)+(0,@S1@)$) {Bayesian Causality};
\node[rotate=-90,text=gold,font=\fontsize{12}{14}\selectfont] at ($(@S0@,0)!0.5!(@F0@,0)+(0,@S2@)$) {Polson \quad Sokolov};
% back cover
\begin{scope}[shift={(@B0@,0)}]
  \node[anchor=north west,text width=@BW@in,text=pale,font=\fontsize{10.5}{14.2}\selectfont,align=left] at (@BX@,@BY@) {\raggedright%
  {\fontsize{15}{19}\selectfont\color{gold} Fit the model. Report the posterior.\\ Then do the accounting.\par}
  \vspace{12pt}
  Double machine learning has become the default way to estimate causal effects with many covariates. It fixes a target, builds an orthogonal score, splits the sample and plugs in black box learners. This book argues for a different order of operations. Model the outcome and the assignment as carefully as the data allow, with priors built for high dimensions, and compute any average treatment effect afterwards as an accounting identity of one posterior.\par
  \vspace{8pt}
  The argument runs from foundations to practice. It sets the frameworks of Fisher, Rubin, Pearl, Robins and Dawid side by side, and covers exchangeability and potential outcomes, the Lanarkshire milk experiment, the theory and failure modes of double machine learning, inverse weighting from Horvitz and Thompson to H\'ajek and the Robins and Ritov example, the horseshoe prior and what it does about regularization induced confounding, BART and Bayesian causal forests, distributional effects, instruments from Wright's tariff study to quarter of birth, the Tian and Pearl bounds on probabilities of causation, and Cornfield's sensitivity analysis of smoking and lung cancer.\par
  \vspace{8pt}
  Every idea is worked on data, real wherever possible: 401(k) eligibility, the LaLonde training programme, the NHEFS smoking cessation cohort, an HIV incentive experiment in Malawi, a horseshoe difference in differences estimator for staggered policy adoption, and a Bayesian merger simulation in which the form of competition is itself uncertain. Generative Bayesian computation turns each posterior into decisions, including risk sensitive ones. Appendices give the mathematical background, the proofs, solutions to most of the exercises and a map to the released code, and a closing chapter sets out open problems.\par
  \vspace{14pt}
  {\color{gold}Nick Polson} is the Robert Law Jr. Professor of Econometrics and Statistics at the Booth School of Business, University of Chicago. {\color{gold}Vadim Sokolov} is Associate Professor in Systems Engineering and Operations Research at George Mason University.\par};
  % barcode area, kept clear for KDP
  \fill[white] (@BC0@,0.375) rectangle (@BC1@,1.575);
\end{scope}
\end{scope}
\end{tikzpicture}
\end{document}
"""
for k,v in dict(W=W,H=H,S0=S0,F0=F0,B0=B0,FX=FX,OY=OY,S1=S1,S2=S2,BW=BW,BX=BX,BY=BY,BC0=BC0,BC1=BC1).items(): tex=tex.replace("@"+k+"@",f"{v:.4f}")
open('cover.tex','w').write(tex)
print(f"spine {s:.4f} in, cover {W:.4f} x {H} in")
