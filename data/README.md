# Data

The real data sets the scripts read. All nine files are stored in this folder, and no script downloads data: the `causaldata` module in the repository root is a short stub that reads these files (`PYTHONPATH=.` makes it importable).

## Licence

The text, figures and PDF of the book are released under CC BY 4.0, and the code under the MIT licence. Those licences do NOT cover the files in this folder. Each data set stays under the terms of its own source, given below. Data released under CC BY-NC 2.0 are for non-commercial use only, so do not reuse them in a commercial product or service. Where the terms are not stated by the source, check before reuse.

| File | Used in | Source | Terms | Credit asked for | Copy or fetch |
|---|---|---|---|---|---|
| `sipp1991.dta` | Intro, Ch. 5, 7, 9, 10, 13, 16 | 1991 Survey of Income and Program Participation (U.S. Census Bureau), extract of Chernozhukov and Hansen (2004), from github.com/VC2015/DMLonGitHub | MIT licence (as distributed with the DML literature and the `DoubleML` package) | Cite Chernozhukov and Hansen (2004) and keep the MIT notice | Copy in this folder |
| `nsw_mixtape.dta`, `cps_mixtape.dta` | Ch. 8 (NSW only), 16 | LaLonde (1986) NSW sample and CPS comparison group as prepared by Dehejia and Wahba (1999, 2002), https://users.nber.org/~rdehejia/nswdata2.html; also in the `causaldata` package | CC BY-NC 2.0, attributable non-commercial use | Cite LaLonde (1986) and Dehejia and Wahba (1999, 2002) | Copy in this folder |
| `psid_controls.dta` | Ch. 16 (solution of Exercise `ex:chlalonde:1`, Appendix D) | Dehejia's PSID comparison group (2,490 men), as prepared by Dehejia and Wahba (1999, 2002), https://users.nber.org/~rdehejia/data/psid_controls.dta, same source and terms as the NSW and CPS row | CC BY-NC 2.0, attributable non-commercial use | Cite LaLonde (1986) and Dehejia and Wahba (1999, 2002) | Copy in this folder |
| `nhefs_complete.dta` | Ch. 17 | NHANES I Epidemiologic Follow-up Study (National Center for Health Statistics), complete case extract of Hernán and Robins (2020); also in the `causaldata` package | The NCHS data user agreement applies | Cite Hernán and Robins (2020) and the NCHS as source | Copy in this folder |
| `thornton_hiv.dta` | Ch. 17 | Thornton (2008), seven variable extract of the replication data (doi 10.3886/E113268V1); also in the `causaldata` package | No licence found on the openICPSR or AEA pages we could read (searched 2 November 2026) | Cite Thornton (2008) and the replication data, openICPSR 113268 | Copy in this folder |
| `castle.dta` | Ch. 15 | Cheng and Hoekstra (2013), built from the FBI Uniform Crime Reports; also in the `causaldata` package | No licence found for the data file; the Mixtape repository that distributes a copy is under an MIT licence, which may not extend to data (searched 2 November 2026) | Cite Cheng and Hoekstra (2013) | Copy in this folder |
| `PNAS_data_2011.csv` | Ch. 1 | Enikolopov, Korovkin, Petrova, Sonin and Zakharov (2013), replication data of the Moscow observers experiment, one row per polling station (3,164 stations, 4 December 2011 Duma election) | Not stated in the file or its source | Cite Enikolopov et al. (2013) | Copy in this folder |
| `card.csv` | Ch. 12 | Card (1995), 3,010 men from the 1976 wave of the National Longitudinal Survey of Young Men, as distributed with Wooldridge (2020) | No licence found; David Card's data page gives only a copyright footer of the University of California (searched 2 November 2026) | Cite Card (1995) and Wooldridge (2020) | Copy in this folder |

## Notes

* The credit column names the works to cite, which is the usual practice for these sources. Only the MIT notice and the CC BY-NC 2.0 attribution are conditions stated by the sources.
* The full references are in the bibliography of the book (`book_tex/chapters/refs.tex`).
* Chapter numbers are the printed ones. Table `tab:data` in Appendix F (`book_tex/chapters/appCode.tex`) lists the same files.
