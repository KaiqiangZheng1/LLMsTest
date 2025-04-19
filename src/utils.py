import json
import base64

class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(data, filename):
    with open(filename, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


# Function to generate base64-encoded frames

def base64_frames_generator(file_paths):
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            yield encoded_image

