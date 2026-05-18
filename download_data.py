import os
import urllib.request
import zipfile

# Define paths and URL
url = "https://zenodo.org/records/11401368/files/DIAS.zip?download=1"
zip_path = "DIAS.zip"
extract_dir = "d_data"

def download_and_extract():
    print(f"Downloading DIAS dataset from Zenodo...")
    # Download the file
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")

    print(f"Extracting dataset to '{extract_dir}'...")
    # Create the directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print("Extraction complete.")

    # Delete the zip file to save space
    print("Cleaning up temporary zip file...")
    os.remove(zip_path)
    
    print("Success! The dataset is ready to use.")

if __name__ == "__main__":
    download_and_extract()