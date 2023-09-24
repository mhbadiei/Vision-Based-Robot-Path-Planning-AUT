from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = 89
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)
    print(hsize)

for i in range(0, 100):
    resizeImage("Dataset/UserDateset/right_" + str(i) + '.png')


