import random
import urllib
import os


# Download data from urls
def download_from_url():

    def downloader(image_url, myPath):
        file_name = random.randrange(1, 10000)
        full_file_name = str(file_name) + '.jpg'
        fullfilename = os.path.join(myPath, full_file_name)
        urllib.request.urlretrieve(image_url, fullfilename)

        with open('data/children/children_path.txt', 'a') as file:
            name = str(fullfilename) + '\n'
            file.write(name)

    urls = list()
    with open("data/children/child_urls.txt", "r") as file:
        urls = file.readlines()

    myPath = 'data/children/child_images/'
    for image_url in urls:
        downloader(image_url, myPath)