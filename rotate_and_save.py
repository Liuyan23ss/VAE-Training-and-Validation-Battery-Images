import os
from PIL import Image

def rotate_images(folder_path):
    if not os.path.exists(folder_path):
        print("資料夾不存在!")
        return
    
    save_folder = os.path.join(folder_path, "rotated")
    os.makedirs(save_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            
            # 原始圖片
            img.save(os.path.join(save_folder, filename))
            
            # 旋轉並另存
            for angle in [90, 180, 270]:
                rotated_img = img.rotate(angle, expand=True)
                new_filename = f"{os.path.splitext(filename)[0]}_{angle}.png"
                rotated_img.save(os.path.join(save_folder, new_filename))
                print(f"已儲存: {new_filename}")
    
    print("所有圖片處理完成！")

# 指定圖片所在資料夾
folder_path = "./ori_rotate"  # 替換為你的圖片資料夾路徑
rotate_images(folder_path)