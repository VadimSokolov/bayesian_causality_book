"""Minimal local stand in for the causaldata package (Huntington Klein, MIT licence).

It loads the data sets used in the book from ../data so that the scripts run without
installing causaldata. Usage matches the package: nsw_mixtape.load_pandas().data
"""
import os, types
import pandas as pd

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def _make(name):
    mod = types.ModuleType(name)
    def load_pandas():
        return types.SimpleNamespace(data=pd.read_stata(os.path.join(_DATA, name + ".dta")))
    mod.load_pandas = load_pandas
    return mod

nsw_mixtape = _make("nsw_mixtape")
cps_mixtape = _make("cps_mixtape")
nhefs_complete = _make("nhefs_complete")
thornton_hiv = _make("thornton_hiv")
castle = _make("castle")
