# src/pdk/layers.py

"""GDS layer definitions.

Each layer is a tuple:
    (layer_number, datatype)
"""

WG = (1, 0)      # full-etch silicon waveguide
SLAB = (2, 0)    # partial-etch slab layer, used later
PORT = (1, 10)   # optional optical port marker
TEXT = (10, 0)   # text labels
