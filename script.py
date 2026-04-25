import os
from PIL import Image

def convert_folder_to_grayscale(input_folder, output_folder):
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created output folder: {output_folder}")

    # 2. Grab all the valid image files
    valid_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"⚠️ No images found in {input_folder}!")
        return

    print(f"🔄 Starting conversion of {len(files)} images...\n")

    # 3. Process each image
    success_count = 0
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Open the image, convert to Grayscale ("L" mode), and save
            with Image.open(input_path) as img:
                grayscale_img = img.convert("L")
                grayscale_img.save(output_path)
            
            success_count += 1
            print(f"✅ Converted: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")

    print(f"\n🎉 Done! Successfully converted {success_count}/{len(files)} images.")
    print(f"📂 You can find them in: {output_folder}")

if __name__ == "__main__":
    # 🔥 UPDATE THESE TWO PATHS 🔥
    # Point this to your folder of color images
    SOURCE_DIR = "./colorimages" 
    
    # The script will create this folder and put the B&W images inside
    DESTINATION_DIR = "./test_bw_images" 
    
    convert_folder_to_grayscale(SOURCE_DIR, DESTINATION_DIR)