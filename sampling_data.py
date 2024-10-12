import os
from osgeo import gdal
import rasterio
from typing import List

# Define constants
TILE_SIZE_X = 256
TILE_SIZE_Y = 256
CHECK = True

# Function to calculate tile size based on image size
def calculate_tile_stamp(size: int) -> int:
    """Determine the tile stamp size based on the total pixel count."""
    if size < 25_000_000:
        return 256
    elif size < 100_000_000:
        return 512
    else:
        return 1024

# Function to create tiles from an input image
def create_tiles(input_path: str, input_filename: str, output_path: str, output_filename_prefix: str) -> None:
    """Creates image tiles from the input image using GDAL."""
    ds = gdal.Open(input_path + input_filename)
    tmp = rasterio.open(input_path + input_filename)
    img = tmp.read()
    
    # Determine tile size based on image dimensions
    size = img.shape[1] * img.shape[2]
    tile_stamp = calculate_tile_stamp(size)

    # Create tiles
    for i in range(0, img.shape[1], tile_stamp):
        for j in range(0, img.shape[2], tile_stamp):
            output_filename = f"{output_path}{output_filename_prefix}{i}_{j}.tif"
            gdal_command = (
                f"gdal_translate -of GTIFF -tr 1 -1 -srcwin {i}, {j}, {TILE_SIZE_X}, {TILE_SIZE_Y} "
                f"{input_path}{input_filename} {output_filename}"
            )
            os.system(gdal_command)

# Function to validate and clean up generated tiles
def validate_tiles(data_path: str, tile_size: int = 256) -> None:
    """Validates generated tiles and removes tiles with negative values or excessive zero values."""
    data_files = [os.path.join(dirname, filename) 
                  for dirname, _, filenames in os.walk(data_path) 
                  for filename in filenames]

    print(f"Total tiles found: {len(data_files)}")

    for i, tile_path in enumerate(data_files):
        with rasterio.open(tile_path) as tmp:
            img = tmp.read()

            # Check for negative values
            if (img < 0).any():
                os.remove(tile_path)
                print(f"Removed tile with negative values: {tile_path}")
                continue

            # Check for tiles with mostly zero values
            zero_count = (img == 0).sum()
            if zero_count > (tile_size ** 2 - 1000):
                os.remove(tile_path)
                print(f"Removed tile with too many zero values: {tile_path}")

        # Print progress
        print(f"Progress: {i + 1}/{len(data_files)} ({(i + 1) / len(data_files) * 100:.2f}%)")

# Main execution
def main():
    input_path = './Crop/'
    output_path = './predict/'
    os.makedirs(output_path, exist_ok=True)

    # Create tiles for specified input files
    for k in [1, 2]:
        input_filename = f'Meuse_{k}m.tif'
        output_filename_prefix = f'tile_meuse_{k}_'
        create_tiles(input_path, input_filename, output_path, output_filename_prefix)
    
    print("Tile creation finished")

    # Validate and clean up tiles if CHECK is enabled
    if CHECK:
        validate_tiles(output_path)

if __name__ == "__main__":
    main()
