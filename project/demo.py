import image_zoomx

image_zoomx.image_predict("images/*.png", 1.5, "output/")
image_zoomx.image_client("PAI", "images/*.png", 2.5, "output/image")
image_zoomx.image_server("PAI")

image_zoomx.video_predict("/home/dell/noise.mp4", 2, "output/predict.mp4")
image_zoomx.video_client("PAI", "/home/dell/noise.mp4", 2, "output/server.mp4")
image_zoomx.video_server("PAI")
