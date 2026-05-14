import cv2
from pathlib import Path

imgs = sorted(Path('F:/ANTS/VisDrone_Dataset/VisDrone2019-DET-val/images').glob('*.jpg'))[:300]
first = cv2.imread(str(imgs[0]))
h, w = first.shape[:2]

writer = cv2.VideoWriter('F:/ANTS/test_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
for i, p in enumerate(imgs):
    f = cv2.imread(str(p))
    writer.write(f)
    print(f"Added frame {i+1}/100")

writer.release()
print("Video created at F:/ANTS/test_video.mp4")