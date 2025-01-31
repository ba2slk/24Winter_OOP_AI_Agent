import os
import json
import sys


script_path = os.path.dirname(os.path.realpath(__file__))


def generate_image_index(image_path, image_index_path):
    image_dir = os.fsencode(image_path)
    image_files = os.listdir(image_dir)
    # image_files = sorted(image_files, key=lambda x: os.fsdecode(x).lower())
    index_result = {}
    
    for i in range(len(image_files)):
        filename = os.fsdecode(image_files[i])
        if filename.endswith(".jpg") or filename.endswith(".JPEG"):
            index_result[i] = filename 

    with open(image_index_path, 'w') as json_file:
        json.dump(index_result, json_file, indent=4)

    print("Successfully created image_index.json")
        

def get_image_path_by_index(index, image_index_path):
    with open(image_index_path, 'r') as json_file:
        index_data = json.load(json_file)
    
    return index_data[index]
        

if __name__=="__main__":
    IMAGE_PATH = os.path.join(script_path, "../data/flickr30k/Images")
    IMAGE_INDEX_PATH = os.path.join(script_path, '../data/image_index.json')

    generate_image_index(IMAGE_PATH, IMAGE_INDEX_PATH)
    print(get_image_path_by_index('300', IMAGE_INDEX_PATH))