from typing import TypedDict, Optional
from PIL import Image
from PIL.Image import Image as ImageType

class ResizedImageData(TypedDict):
    image: ImageType
    resized_image: ImageType
    new_width: int
    new_height: int
    top_left_x: int
    top_left_y: int

def resize_fill(
    image: Image.Image,
    target_width: int,
    target_height: int,
    resample_mode: Image.Resampling = Image.Resampling.LANCZOS,
):
    width, height = image.size

    width_ratio = target_width / width
    height_ratio = target_height / height

    resize_ratio = min(width_ratio, height_ratio)

    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    resized_image = image.resize((new_width, new_height), resample_mode)

    top_left_x = (target_width - new_width) // 2
    top_left_y = (target_height - new_height) // 2

    final_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    final_image.paste(resized_image, (top_left_x, top_left_y))

    return ResizedImageData(
        image=image,
        resized_image=final_image,
        new_width=new_width,
        new_height=new_height,
        top_left_x=top_left_x,
        top_left_y=top_left_y
    )

def resize_fill_restore(
    resized_data: ResizedImageData,
    new_image: Optional[Image.Image] = None,
    resample_mode: Image.Resampling = Image.Resampling.LANCZOS,
):
    image = resized_data['image']
    resized_image = new_image or resized_data['resized_image']
    new_width = resized_data['new_width']
    new_height = resized_data['new_height']
    top_left_x = resized_data['top_left_x']
    top_left_y = resized_data['top_left_y']

    original_width, original_height = image.size

    cropped_resized_image = resized_image.crop((
        top_left_x,
        top_left_y,
        top_left_x + new_width,
        top_left_y + new_height
    ))

    restored_image = cropped_resized_image.resize((original_width, original_height), resample_mode)
    return restored_image
