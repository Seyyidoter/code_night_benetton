## Current strategy
This repository currently uses an explainable hybrid strategy:
- SMA trend filter
- OBV volume confirmation
- Bollinger-based overextension filter
- volatility guard
- fixed TP/SL with 1:2 risk-reward structure

The implementation is built on top of `cnlib`.