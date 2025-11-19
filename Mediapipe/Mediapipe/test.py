from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (400, 200), (255, 255, 255))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 40)
draw.text((10, 50), "đ â ê ă ô ư ấ ^ ' `", fill=(0, 0, 0), font=font)
img.show()
