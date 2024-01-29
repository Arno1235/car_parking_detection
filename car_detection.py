import sys
import os
import cv2
import torch
from telethon import TelegramClient, events
from picamera2 import Picamera2


model = torch.hub.load("ultralytics/yolov5", "yolov5s")
bot = TelegramClient('bot', secret, secret).start(bot_token=secret)

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.start()

def remove_previous_detection():
    if os.path.exists("out"):
        os.system("rm -rf " + "out")

def get_image():
    picam2.capture_file("picam.jpg")
    return "picam.jpg"

def run_model(img):

    results = model(img)

    results.print()
    results.save(save_dir='out')

    return results

def detect_location(predictions, img):

    response = ""

    labels, cord_thres = predictions.xyxyn[0][:, -1].cpu().numpy(), predictions.xyxyn[0][:, :-1].cpu().numpy()

    im_w, im_h = 1024, 768

    # name, label, x, y
    coordinates = [
        ['A0', 2, 214/im_w, 517/im_h],
        ['A1', 2, 386/im_w, 433/im_h],
        ['A2', 2, 485/im_w, 382/im_h],
    ]

    for coordinate in coordinates:
        n, l, x, y = coordinate

        for lab, coo in zip(labels, cord_thres):

            if l == lab:
                x1, y1, x2, y2, conf = coo

                if x > x1 and x < x2 and y > y1 and y < y2:
                    response += n + " occupied\n"
                    img = cv2.circle(img, (int(x * im_w), int(y * im_h)), 8, (0, 0, 255), -1)
                    break
       
        else:
            response += n + " free\n"
            img = cv2.circle(img, (int(x * im_w), int(y * im_h)), 8, (0, 255, 0), -1)
   
    return response[:-1], img


@bot.on(events.NewMessage(pattern='/run'))
async def run(event):
    print("/run")

    remove_previous_detection()
    predictions = run_model(get_image())

    img = cv2.imread('out/picam.jpg')
    response, img = detect_location(predictions, img)
    cv2.imwrite('out/result.jpg', img)

    await event.reply(response)
    await event.respond(file=open('out/result.jpg', 'rb'))

    raise events.StopPropagation

@bot.on(events.NewMessage(pattern='/image'))
async def image(event):
    print("/image")

    await event.reply(file=open(get_image(), 'rb'))

    raise events.StopPropagation

@bot.on(events.NewMessage(pattern='/status'))
async def status(event):
    print("/status")
    await event.reply("standby")
    raise events.StopPropagation

@bot.on(events.NewMessage(pattern='/stop'))
async def status(event):
    print("/stop")
    sys.exit()
    raise events.StopPropagation


def main():
    print("ready")
    bot.run_until_disconnected()

if __name__ == "__main__":
    main()
