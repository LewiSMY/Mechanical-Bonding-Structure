import gdown
import tarfile
import os

os.chdir('../results')
url = 'https://drive.google.com/drive/folders/14tUD6OP2ZKzp_M64qTQty5Ov2nFCco_v?usp=drive_link'

output = 'data.tgz'
gdown.download(url, output, quiet=False)

tar = tarfile.open(output, "r:")
tar.extractall()
tar.close()

os.chdir('../src')