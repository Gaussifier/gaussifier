# Renderer device helpers

`config.h` and `helpers.cuh` are the tile and projection helpers of the Splat2D simple-sum
rasterizer that the fused K-loop renderer (`../../kloop_render.cu`) builds on. They are kept here
so the fused runtime builds on its own. `third_party/glm` is
[OpenGL Mathematics](https://github.com/g-truc/glm) (MIT, see `third_party/glm/copying.txt`).
