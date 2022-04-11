import psutil
import os
import zipfile
import tempfile

def show_memory_info():
    process = psutil.Process(os.getpid())
    print(process.memory_info())

def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)
