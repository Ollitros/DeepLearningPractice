from PIL import Image


image = Image.open('data/images/style.jpg')

image = image.resize((640, 640))
image.save("style.JPG")

image = Image.open('data/images/content.jpg')


image = image.resize((640, 640))
image.save("content.JPG")