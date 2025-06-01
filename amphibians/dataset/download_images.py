import os
import csv
import requests
from tqdm import tqdm

# File paths
input_file = 'v2_dataset.csv'
output_directory = 'downloaded_images'
missing_images_file = 'missing_images.csv'

# iNaturalist image base URL (using 'large' size)
base_url = "https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/large.{extension}"

# Taxon ID to species name mapping
amphibien_taxon_ids = {
    'feuersalamander': 27726,
    'alpensalamander': 27735,
    'bergmolch': 135104,
    'kammmolch': 27717,
    'teichmolch': 556656,
    'rotbauchunke': 24495,
    'gelbbauchunke': 556621,
    'knoblauchkröte': 25312,
    'erdkröte': 326296,
    'kreuzkröte': 65450,
    'wechselkröte': 1453013,
    'laubfrosch': 424147,
    'moorfrosch': 25488,
    'springfrosch': 25669,
    'grasfrosch': 25591,
    'wasserfrosch': [66322, 66334, 1558164]  # Merged pool
}

# Reverse mapping {taxon_id: species_name}
taxon_id_to_species = {}
for species, taxon in amphibien_taxon_ids.items():
    if isinstance(taxon, list):  # Handle merged frog taxa
        for t in taxon:
            taxon_id_to_species[t] = species
    else:
        taxon_id_to_species[taxon] = species

# Ensure output directories exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Prepare missing images CSV
missing_entries = []
missing_header_written = False

# Load the dataset and start downloading
with open(input_file, 'r') as file:
    reader = csv.DictReader(file, delimiter='\t')

    # Progress bar setup
    total_rows = sum(1 for _ in open(input_file, 'r')) - 1  # Exclude header row
    file.seek(0)
    next(reader)  # Skip header

    for row in tqdm(reader, total=total_rows, desc="Downloading images"):
        photo_uuid = row['photo_uuid']
        photo_id = row['photo_id']
        taxon_id = int(row['taxon_id'])
        extension = row['extension'].lower()  # Ensure consistent file extensions

        # Get species name
        species_name = taxon_id_to_species.get(taxon_id, "unknown_species")

        # Create species directory if not exists
        species_dir = os.path.join(output_directory, species_name)
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)

        # Construct the image URL
        image_url = base_url.format(photo_id=photo_id, extension=extension)
        image_path = os.path.join(species_dir, f"{photo_uuid}.{extension}")

        # Skip if file already exists (resume support)
        if os.path.exists(image_path):
            continue

        # Download the image
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(image_path, 'wb') as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
            else:
                print(f"Failed to download {image_url} (HTTP {response.status_code})")
                row['status'] = 'deleted'
                missing_entries.append(row)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {image_url}: {e}")
            row['status'] = 'deleted'
            missing_entries.append(row)

# Save missing images to CSV
if missing_entries:
    with open(missing_images_file, 'w', newline='') as missing_file:
        writer = csv.DictWriter(missing_file, fieldnames=list(missing_entries[0].keys()))
        writer.writeheader()
        writer.writerows(missing_entries)

print(f"Image downloading complete. {len(missing_entries)} missing images recorded in '{missing_images_file}'.")
