import xarray as xr
import zarr
import pandas as pd
import numpy as np
import h5netcdf
from pathlib import Path
import glob

def ncs_to_zarr():
    """
    first use this function to convert all the nc files to one zarr
    """
    # Directory where all your daily .nc4 files are
    data_dir = Path("path")

    # Match all daily files (adjust the pattern if needed)
    files = sorted(glob.glob(str(data_dir / "*.nc")))

    # Open and concatenate along the 'time' dimension
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",  # relies on matching dimension names
        engine="netcdf4",
        parallel=True,
        chunks="auto"
    )

    # Optional: inspect result
    print(ds)

    # OR: Save to Zarr
    ds.to_zarr("temp2019/uk_albedo.zarr", mode="w", zarr_format=2)


def combine_zarrs():
    """
    if using variables from differnet zarrs, use this function to combine them
    """
    # Open the two Zarr datasets
    ds1 = xr.open_zarr("path1.zatt", consolidated=True)
    ds2 = xr.open_zarr("path2.zarr", consolidated=True)

    # Merge them on common coordinates (time, lat, lon)
    # This assumes they are aligned on those dimensions or can be aligned
    merged_ds = xr.merge([ds1, ds2], compat="no_conflicts")

    # Save the merged dataset to a new Zarr file
    merged_ds.to_zarr("output.zarr", mode="w", consolidated=True, zarr_format=2)


def combine_variables():
    """
    this function is used to to merger multiple variables into one
    """
    # Open original Zarr
    ds = xr.open_zarr("input_path.zarr", consolidated=True)

    # Variables to combine (excluding TOTSCATAU)
    var_names = ["ALBEDO", "DUSMASS", "TOTANGSTR", "TOTEXTTAU"]

    # Combine into new 'feature' dimension
    combined = xr.concat([ds[var] for var in var_names], dim="feature")

    # Add the feature labels
    combined = combined.assign_coords(feature=var_names)

    # Drop all original individual variables, including TOTSCATAU
    ds = ds.drop_vars(var_names)

    # Add the combined variable
    ds["combined_variable"] = combined

    # Clear any problematic encodings
    for var in ds.variables:
        ds[var].encoding.clear()

    # Save to new Zarr
    ds.to_zarr("output_path.zarr", mode="w", zarr_format=2)

    print(ds)



def mean_and_std():
    """
    this function is used to calculate mean and standard devidation for the variables in the data
    """
    ds = xr.open_zarr("input_path.zarr")  # chunks={} to preserve dask lazy loading
    # Mask out large fill values (optional but recommended)
    #fill_value_threshold = 1e10
    data = ds['combined_variable']
    # Calculate mean and std dev across time, lat, and lon for each feature
    mean_per_feature = data.mean(dim=["time", "lat", "lon"])
    std_per_feature = data.std(dim=["time", "lat", "lon"])

    # Get feature names
    feature_names = ds['feature'].values

    # Evaluate and format output
    print("Feature-wise Mean:")
    for feature in feature_names:
        mean_val = mean_per_feature.sel(feature=feature).values.item()
        print(f"{feature}: {mean_val:.10f}")

    print("\nFeature-wise Standard Deviation:")
    for feature in feature_names:
        std_val = std_per_feature.sel(feature=feature).values.item()
        print(f"{feature}: {std_val:.10f}")


def change_hours():
    """
    merra data originally has half hour time stamps like 1:30,2:30,3:30
    which currently is not supported by data sampler, this function will change the
    data and round it up to the hour
    """
    # Open your Zarr dataset
    ds = xr.open_zarr("input_path.zarr")

    # Round the time coordinate up to the next hour
    rounded_time = pd.to_datetime(ds.time.values).ceil('H')

    # Assign the updated time coordinate back to the dataset
    ds = ds.assign_coords(time=("time", rounded_time))

    # (Optional) Save to new Zarr or overwrite
    ds.to_zarr("output_path.zarr", mode="w")  # Use mode="w" to overwrite, or choose a new path
    ds.to_netcdf("output_path.nc")

    print(ds)
