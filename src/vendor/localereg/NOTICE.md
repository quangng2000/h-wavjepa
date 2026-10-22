`losses.py` is copied without mathematical changes from
https://github.com/alessioborgi/LocaLeJEPA/blob/771361c18d37e65df3dc188761ffb26751ce4e37/src/losses.py.

The upstream MIT license is retained at `third_party/licenses/LocaLeJEPA.txt`.
The audio adapter is `src/objectives/localereg.py`. It calls the upstream torch
k-means backend separately per time step; audio time steps are not augmented views.
The adapter uses 256 quadrature nodes for the 768-dimensional target, instead of
the upstream function default of 64. All adaptations are recorded in checkpoint metadata.
