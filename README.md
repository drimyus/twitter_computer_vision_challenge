
# Twitter Computer Vision Challenge
Taking the above images as the training set, output the coordinates for the bounding boxes contain the twitter “following button”

## Installation

First, with python and pip3 installed, install the scripts requirements:

```bash
pip3 install -r requirements.txt
```
Then you must compile the Protobuf libraries:

```bash
sudo apt install protobuf-compiler
protoc object_detection/protos/*.proto --python_out=.
```

Add `models` and `models/slim` to your `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

>_**Note:** This must be ran every time you open terminal, or added to your `~/.bashrc` file.


## Usage
### 1) Create the TensorFlow Records
Run the script:

```bash
python3 object_detection/create_tf_record.py
```

Once the script finishes running, you will end up with a `train.record` and a `val.record` file. This is what we will use to train the model.

### 2) Download a Base Model
You can find models to download from this [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Each model varies in accuracy and speed. I used `ssd_inception_v2_coco` for the demo.

Extract the files and move all the `model.ckpt` to our models directory.

>_**Note:** If you don't use `faster_rcnn_resnet50`, replace `faster_rcnn_resnet50.config` with the corresponding config file.

### 3) Train the Model
Run the following script to train the model:

```bash
python3 object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=models/ssd_inception_v2_coco.config
```


### 4) Export the Inference Graph
You can find checkpoints for your model in `train`.
- `model.ckpt-STEP_NUMBER.data-00000-of-00001`
- `model.ckpt-STEP_NUMBER.index`
- `model.ckpt-STEP_NUMBER.meta`

In order to use the model, you first need to convert the checkpoint files (`model.ckpt-STEP_NUMBER.*`) into a frozen inference graph by running this command:

```bash
python3 object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path models/ssd_inception_v2_coco.config \
        --trained_checkpoint_prefix train/model.ckpt-NUMBER \
        --output_directory output_inference_graph
```

You should see a new `output_inference_graph` directory with a `frozen_inference_graph.pb` file.

### 5) Test the Model
Just run the following command:

```bash
python3 object_detection/object_detection_runner.py
```

It will run your object detection model found at `output_inference_graph/frozen_inference_graph.pb` on all the images in the `test_images` directory and output the results in the `output/test_images` directory.

## Results
Here’s what I got from running my model over testing image
![](test/output/result.jpg)
