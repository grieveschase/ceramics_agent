import os
import requests
from pathlib import Path
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_images(case_id: str, image_urls: list[str]) -> Path:
    '''
    Download images from the internet and save them to a local directory.
    Args:
        case_id: The case ID.
        image_urls: A list of image URLs.
    Returns:
        img_dir: The local directory where the images were saved.
    '''
    image_directory = rf".\images\{case_id}"
        
    img_dir = Path(image_directory)
    img_dir.mkdir(parents=True, exist_ok=True)

    for image_url in image_urls:
        image_path = os.path.join(image_directory, os.path.basename(image_url))
        response = requests.get(image_url, verify=False)
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        

    return img_dir



## RookWood Vase
case_id="12345"
image_urls=[
    r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32728000.jpg",
    r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32727999.jpg",
    r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32728001.jpg",
    r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32727998.jpg",  # Bottom Image.

    ]
img_dir:Path = download_images(case_id, image_urls)

print(img_dir)