Fabric Algorithm Settings Explained

Page & Output Settings

output_width_inches (default: 4.5)
The width of the final output page in inches.
This defines the physical paper width the plotter draws on.

output_height_inches (default: 6.0)
The height of the final output page in inches.
Together with width, this defines the canvas size.

margin_inches (default: 0.125)
The blank border around all edges of the drawing.
Prevents lines from touching the paper edge (1/8 inch ≈ 3 mm).

calc_dpi (default: 300)
Calculation resolution used for internal math.
Higher values increase precision but slow generation.
300 is a good balance for most use cases.

Grid & Thread Density

grid_cols (default: 150)
Number of vertical "threads" running top to bottom.
More columns increase horizontal detail and density.

grid_rows (default: 200)
Number of horizontal "threads" running left to right.
More rows produce a denser vertical pattern.
Each row becomes one continuous flowing line.

Noise Pattern Controls
(Controls wrinkles and folds)

noise_scale_x (default: 0.002)
Frequency of horizontal wrinkles.
Smaller values create wider, stretched folds.
Larger values create tighter, compressed wrinkles.

noise_scale_y (default: 0.002)
Frequency of vertical wrinkles.
Smaller values create taller, smoother waves.
Larger values create shorter, more frequent undulations.

noise_octaves (default: 4)
Number of noise detail layers.
More octaves produce more complex, natural textures.
Typical range: 1–8 (higher is slower).

noise_persistence (default: 0.5)
Contribution strength of each noise layer.
Higher values (0.7–0.9) create rougher textures.
Lower values (0.3–0.5) create smoother fabric.

noise_lacunarity (default: 2.0)
Frequency multiplier between noise layers.
2.0 is standard.
Higher values add finer details faster.

Displacement & Effect

displacement_magnitude (default: 500)
Maximum distance threads can move from their grid position.
Higher values create deeper folds and dramatic waves.
Lower values create subtle ripples.
Units are in pixels at calc_dpi resolution.

enable_weave (default: true)
Currently unused.
Originally intended for over/under weave effects.
Disabled to ensure continuous plotter lines.

Summary

The fabric algorithm generates a grid of horizontal threads and uses Perlin noise to displace them.
This creates the appearance of wrinkled or folded fabric.
You control thread density, wrinkle frequency, and displacement strength.
