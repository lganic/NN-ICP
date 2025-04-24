import os
import shutil
import random
import uuid
from tqdm import tqdm
from PIL import Image

source_dir = '/blue/aosmith1/logan.boehm/texture_dataset'
output_dir = '/blue/aosmith1/logan.boehm/processed_texture_dataset'

train_ratio = .8
val_ratio = .1

SIZE = 256


# Clear old data if it exists
if os.path.exists(output_dir):
    print('Removing old data...')
    shutil.rmtree(output_dir)

# First, generate a list of all the png files in the directory
all_files = []

for dir, _, files in os.walk(source_dir):

    for file in files:

        if not file.endswith('.png'):
            continue # Not sure why this would happen, these files should have been removed by now

        full_path = os.path.join(dir, file)

        all_files.append(full_path)

# Shuffle the samples
random.shuffle(all_files)

# Create test/train/val split
num_samples = len(all_files)

train_val_ratio = train_ratio + val_ratio

train_index = int(train_ratio * num_samples)
val_index = int(train_val_ratio * num_samples)

training_samples = all_files[:train_index]
val_samples = all_files[train_index: val_index]
test_samples = all_files[val_index:]

def process_samples(group, group_name):

    target_directory = os.path.join(output_dir, group_name)
    os.makedirs(target_directory, exist_ok=True)

    for file in tqdm(group, desc = f'Processing: {group_name}'):

        try:
            img = Image.open(file)
        except Exception as e:
            print(f'Error loading image: {file}')
            print(e)
            continue
        resized_img = img.resize((SIZE, SIZE))

        new_filename = str(uuid.uuid4()) + '.png'
        new_path = os.path.join(target_directory, new_filename)

        resized_img.save(new_path)

process_samples(training_samples, 'train')
process_samples(val_samples, 'val')
process_samples(test_samples, 'test')