from gdown import download
import subprocess
import os

subprocess.run("rm -Rf data", shell=True)
os.mkdir("data")

download("https://drive.google.com/uc?id=1_rmpEwhBAGfT_De3tYaCOuGJwEHN7JlC")
subprocess.run("mv aochildes.zip data", shell=True)
subprocess.run(
    "cd data; unzip aochildes.zip; rm -Rf __MACOSX aochildes.zip", shell=True)
