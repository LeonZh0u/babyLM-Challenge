from gdown import download
import subprocess
import os

subprocess.run("rm -Rf data", shell=True)
os.mkdir("data")

download("https://drive.google.com/uc?id=15AMwAEt0QM9IBa1pVRNrd9O-B4xN_59-")
subprocess.run("mv babylm_data.zip data", shell=True)
subprocess.run(
    "cd data; unzip babylm_data.zip; rm -Rf __MACOSX babylm_data.zip", shell=True)
