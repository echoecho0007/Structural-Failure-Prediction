"""Minimal Streamlit stand-in for smoke-testing app.py (records calls, returns widget defaults)."""
import sys
LOG=[]
class _Stop(Exception): pass
def _rec(name):
    def f(*a, **k):
        LOG.append((name, a[0] if a else None)); 
        return _Ctx()
    return f
class _Ctx:
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def __getattr__(self, n): return _rec(n)
def number_input(label, min_value=None, value=None, **k): LOG.append(("number_input",label,value)); return value
def selectbox(label, options, index=0, **k): return options[index]
def columns(n): return [_Ctx() for _ in range(n)]
def cache_resource(f): return f
def stop(): raise _Stop()
sidebar=_Ctx(); 
def expander(*a,**k): return _Ctx()
def __getattr__(n): return _rec(n)
