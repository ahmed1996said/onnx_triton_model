import tritonclient.http as httpclient
import numpy as np

TRITON_SERVER_URL = 'localhost:8000'

def main():
    # Create a random image array
    image_array = np.random.uniform(low=-1.0, high=1.0, size=(3,512,512)).astype(np.float32)
    image_array = image_array[None]  # (1, 3, 512, 512), float32

    # prepare the input for http request
    infer_input = httpclient.InferInput(
        "input", image_array.shape, datatype="FP32"
    )
    infer_input.set_data_from_numpy(image_array)

    # initialize the inference server client
    client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

    # make inference request
    classification_response = client.infer(
        model_name="classifier", inputs=[infer_input]
    )

    # fetch inference results
    logits = classification_response.as_numpy("output")

    print(f"Recieved logits: {logits}")

if __name__ == "__main__":
    main()