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
    gen(
        "sample.qr.png",
        prompt="1mechanical girl,ultra realistic details, portrait, global illumination, shadows, octane render, 8k, ultra sharp,intricate, ornaments detailed, cold colors, metal, egypician detail, highly intricate details, realistic light, trending on cgsociety, glowing eyes, facing camera, neon details, machanical limbs,blood vessels connected to tubes,mechanical vertebra attaching to back,mechanical cervial attaching to neck,sitting,wires and cables connecting to head",
        
        brightness_image="https://github.com/anotherjesse/dream-templates/assets/27/c5df2f7c-7a0c-43ad-93d6-921af0759190",
        tile_image="https://github.com/anotherjesse/dream-templates/assets/27/c5df2f7c-7a0c-43ad-93d6-921af0759190",
        qr_image="https://github.com/anotherjesse/dream-templates/assets/27/c5df2f7c-7a0c-43ad-93d6-921af0759190",
        depth_image="https://github.com/anotherjesse/dream-templates/assets/27/c5df2f7c-7a0c-43ad-93d6-921af0759190",
        # hed_image="https://github.com/anotherjesse/dream-templates/assets/27/c5df2f7c-7a0c-43ad-93d6-921af0759190",
        qr_conditioning_scale=0.1,
        depth_conditioning_scale=0.4,
        brightness_conditioning_scale=0.3,
        tile_conditioning_scale=0.5,
        width=728,
        height=728,
        # guidance_scale=30,
        # eta=100,
        # seed=42,
        negative_prompt="ugly, disfigured, low quality, blurry",
        # scheduler="K_EULER",
        # guess_mode=True,
        steps=20,
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
