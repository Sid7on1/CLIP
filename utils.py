def load_image(image_path, transform):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return transform(img)
