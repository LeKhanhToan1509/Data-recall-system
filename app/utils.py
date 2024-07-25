import os, datetime

def get_file_download_date(file_path):
    file_stats = os.stat(file_path)
    access_time = file_stats.st_atime
    download_date = datetime.datetime.fromtimestamp(access_time)
    return str(download_date)

def crop_image(image, crop_box):
    cropped_image = image.crop(crop_box)
    return cropped_image