from PIL import Image


image = Image.open('data/images/style1.jpg')

image = image.resize((449, 493))
image.save("style.JPG")

image = Image.open('data/images/content.jpg')


image = image.resize((449, 493))
image.save("content.JPG")