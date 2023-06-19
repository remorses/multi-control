import base64
import requests
import sys
import os


def gen(output_fn, **kwargs):
    # if os.path.exists(output_fn):
    #     print("Skipping", output_fn)
    #     return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    # gen(
    #     "sample.none.png",
    #     prompt="taylor swift in a mid century modern bedroom",
    #     seed=42,
    #     steps=30,
    # )
    # gen(
    #     "sample.canny.png",
    #     prompt="taylor swift in a mid century modern bedroom",
    #     canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
    #     seed=42,
    #     steps=30,
    # )
    
    code = 'https://ucarecdn.com/a308cfd2-9b1f-4b3c-95ed-3cb2647844eb/'
    w = 1024
    gen(
        "sample.qr.png",
        prompt="houses in the snow, 4k",
        brightness_image=code,
        qr_image=code,
        # tile_image=code,
        # depth_image=code,
        # hed_image=code,
        qr_conditioning_scale=0.5,
        depth_conditioning_scale=0.3,
        brightness_conditioning_scale=0.5,
        # tile_conditioning_scale=1,
        width=w,
        height=w,
        # guidance_scale=30,
        # eta=100,
        # seed=42,
        negative_prompt="ugly, disfigured, low quality, blurry",
        # scheduler="K_EULER",
        # guess_mode=True,
        steps=20,
        num_samples=1,
    )
    return
    gen(
        "sample.canny.guess.png",
        prompt="",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.hough.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.hough.guess.png",
        prompt="",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.normal.png",
        prompt="",
        normal_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.depth.png",
        prompt="",
        depth_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.both.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.both.guess.png",
        prompt="",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.scaled.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        hough_conditioning_scale=0.6,
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        canny_conditioning_scale=0.8,
        seed=42,
        steps=30,
    )
    gen(
        "sample.seg.png",
        prompt="modern bedroom with plants",
        seg_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
    )
    gen(
        "sample.hed.png",
        prompt="modern bedroom with plants",
        hed_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
    )
    gen(
        "sample.pose.png",
        prompt="a man in a suit by van gogh",
        pose_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
        seed=42,
    )
    gen(
        "sample.scribble.png",
        prompt="painting of cjw by van gogh",
        scribble_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_vermeer_scribble.png",
        seed=42,
    )



if __name__ == "__main__":
    main()
