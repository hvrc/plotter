import matplotlib.pyplot as plt
import numpy as np
import noise
import random

def generate_fabric_texture(
    width=800, height=800,
    grid_cols=100, grid_rows=100,
    noise_scale_x=0.02, noise_scale_y=0.02,
    noise_octaves=4, noise_persistence=0.5, noise_lacunarity=2.0,
    displacement_magnitude=30,
    line_color='#333333', line_width=0.5,
    output_filename='fabric_texture.png'
):
    """
    Generates a 2D fabric texture using only lines and saves it to an image file.

    Args:
        width (int): Width of the output image in pixels.
        height (int): Height of the output image in pixels.
        grid_cols (int): Number of vertical "warp" threads.
        grid_rows (int): Number of horizontal "weft" threads.
        noise_scale_x (float): Scale of the noise in the x-direction (lower is smoother).
        noise_scale_y (float): Scale of the noise in the y-direction.
        noise_octaves (int): Number of layers of noise.
        noise_persistence (float): How much each octave contributes to the overall shape.
        noise_lacunarity (float): How much detail is added at each octave.
        displacement_magnitude (float): The maximum amount a point can be displaced.
        line_color (str): Hex color code for the lines.
        line_width (float): Width of the lines.
        output_filename (str): The name of the output image file.
    """

    # 1. Create a base grid of points
    # We create a grid that is slightly larger than the final image to avoid edge effects
    pad = 0.1
    x = np.linspace(-pad * width, width * (1 + pad), grid_cols)
    y = np.linspace(-pad * height, height * (1 + pad), grid_rows)
    X, Y = np.meshgrid(x, y)

    # 2. Generate Perlin noise for displacement
    # We need a separate noise field for x and y displacement to create complex folds.
    # A random seed is added to make the pattern different each time.
    seed_x = random.randint(0, 1000)
    seed_y = random.randint(0, 1000)

    noised_X = np.zeros_like(X)
    noised_Y = np.zeros_like(Y)

    for i in range(grid_rows):
        for j in range(grid_cols):
            # Calculate noise values
            nx = noise.pnoise2(X[i, j] * noise_scale_x, Y[i, j] * noise_scale_y,
                               octaves=noise_octaves,
                               persistence=noise_persistence,
                               lacunarity=noise_lacunarity,
                               repeatx=width, repeaty=height,
                               base=seed_x)
            ny = noise.pnoise2(X[i, j] * noise_scale_x, Y[i, j] * noise_scale_y,
                               octaves=noise_octaves,
                               persistence=noise_persistence,
                               lacunarity=noise_lacunarity,
                               repeatx=width, repeaty=height,
                               base=seed_y)
            
            noised_X[i, j] = nx
            noised_Y[i, j] = ny

    # 3. Apply the displacement to the grid points
    # The magnitude controls how extreme the folds are.
    X_distorted = X + noised_X * displacement_magnitude
    Y_distorted = Y + noised_Y * displacement_magnitude

    # 4. Set up the plot with margin
    # Margin of 0.5 inches. Width/height are in pixels at 100 DPI, so margin = 50 pixels
    margin_inches = 0.5
    figure_dpi = 100  # DPI for figure creation (matches width/height pixel values)
    margin_pixels = margin_inches * figure_dpi  # 0.5 * 100 = 50 pixels
    
    # Total canvas size including margins
    total_width = width + 2 * margin_pixels
    total_height = height + 2 * margin_pixels
    
    fig, ax = plt.subplots(figsize=(total_width/figure_dpi, total_height/figure_dpi), dpi=figure_dpi)
    ax.set_facecolor('white')
    
    # Remove axes
    ax.set_axis_off()
    
    # Set limits to the full canvas (including margin area for drawing)
    ax.set_xlim(-margin_pixels, width + margin_pixels)
    ax.set_ylim(-margin_pixels, height + margin_pixels)
    
    # This is crucial to prevent the image from being stretched
    ax.set_aspect('equal')
    
    # Create a clipping box for the visible area (excluding margins)
    from matplotlib.patches import Rectangle
    clip_box = Rectangle((0, 0), width, height, transform=ax.transData) 

    # 5. Draw the "weft" (horizontal) threads only
    for i in range(grid_rows):
        # Plot a single continuous line for each weft thread
        line = ax.plot(X_distorted[i, :], Y_distorted[i, :],
                color=line_color, linewidth=line_width, alpha=0.8)[0]
        # Clip the line to the visible area (excluding margins)
        line.set_clip_path(clip_box)
        
        # --- Optional: Add a "weave" effect for more realism ---
        # This part is more complex and adds small gaps to simulate threads going under/over.
        # For a simpler, more fluid look, you can comment out this block.
        for j in range(grid_cols - 1):
             # Define the start and end points of a single weft segment
            p1_x, p1_y = X_distorted[i, j], Y_distorted[i, j]
            p2_x, p2_y = X_distorted[i, j+1], Y_distorted[i, j+1]

            # A simple checkerboard pattern to decide if the thread is "over" or "under"
            if (i + j) % 2 == 0:
                # "Over" - draw the full segment
                line = ax.plot([p1_x, p2_x], [p1_y, p2_y], color=line_color, linewidth=line_width)[0]
                line.set_clip_path(clip_box)
            else:
                # "Under" - draw a shorter segment with a gap in the middle
                # Calculate points 25% and 75% along the segment
                mid1_x = p1_x + 0.25 * (p2_x - p1_x)
                mid1_y = p1_y + 0.25 * (p2_y - p1_y)
                mid2_x = p1_x + 0.75 * (p2_x - p1_x)
                mid2_y = p1_y + 0.75 * (p2_y - p1_y)
                # Draw the two short segments
                line1 = ax.plot([p1_x, mid1_x], [p1_y, mid1_y], color=line_color, linewidth=line_width)[0]
                line1.set_clip_path(clip_box)
                line2 = ax.plot([mid2_x, p2_x], [mid2_y, p2_y], color=line_color, linewidth=line_width)[0]
                line2.set_clip_path(clip_box)
        # -------------------------------------------------------

    # 7. Save the final image
    # bbox_inches='tight' and pad_inches=0 remove any extra white space
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Successfully generated fabric texture image: {output_filename}")

if __name__ == '__main__':
    # You can experiment with these parameters to get different looks.
    # - Increase `grid_cols` and `grid_rows` for a finer, more dense fabric.
    # - Increase `displacement_magnitude` for deeper, more dramatic folds.
    # - Adjust `noise_scale_x` and `noise_scale_y` to change the frequency of the folds.
    generate_fabric_texture(
        width=450,  # 4.5 inches at 100 DPI
        height=600,  # 6 inches at 100 DPI
        grid_cols=150,  # Number of vertical threads (still needed for grid)
        grid_rows=200,  # 500 horizontal lines as requested
        noise_scale_x=0.002, # Controls the scale of the folds (lower = smoother, larger waves)
        noise_scale_y=0.002,
        displacement_magnitude=500, # Controls the depth of the folds (increased for even higher amplitude)
        line_width=0.3,
        output_filename='my_fabric.png'
    )