@echo off
cd /d "C:\Users\gbran\Documents\yolo-static-filter"
"C:\Users\gbran\AppData\Local\Programs\Python\Python312\python.exe" run_on_video.py --source vtest.avi --model yolo11n.pt --classes 0 --clip --save clip_out.mp4
pause
