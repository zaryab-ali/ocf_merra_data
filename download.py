import requests
import os


"""
disc.gsfc.nasa.gov give you the data as seperate .nc files if you split/crop the data in any way
each file contains data for a 24 hour period, meaning one file per day
the data will be given to you as a txt file contaning all the links you can iterate through and download

tip : when getting the download linsk, choose  GES DISC subsetter links instead of OPENDAP links, the former is much more 
reliable than the later
"""

token = "YOUR EARTH DATA TOKEN"
txt_file_path = "PATH TO THE TXT CONTAINING ALL THE LINKS"
output_dir = "PATH to the folder where files will be stores"
os.makedirs(output_dir, exist_ok=True)

headers = {
    "Authorization": f"Bearer {token}"
}

with open(txt_file_path, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

for i, url in enumerate(urls, start=443):
    save_path = os.path.join(output_dir, f"merra{i}.nc")

    if os.path.exists(save_path):
        print(f"[SKIP] merra{i}.nc already exists.")
        continue

    print(f"[DOWNLOADING] merra{i}.nc from {url}")
    response = requests.get(url, headers=headers, stream=True, timeout=1200)
    response.raise_for_status()
    print("saving")

    with open(save_path, "wb") as f_out:
        for chunk in response.iter_content(chunk_size=8192):
            f_out.write(chunk)

    print(f"[DONE] merra{i}.nc")
