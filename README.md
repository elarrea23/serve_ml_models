<div align="center">

# Serve ML Models

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=gray"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=black"></a>
<a href="https://pytorch.org/serve/"><img alt="TorchServe" src="https://img.shields.io/badge/-TorchServe-0BB00B?style=for-the-badge&logo=pytorchlightning&logoColor=orange"></a>
<a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/-Docker-24ACD7?style=for-the-badge&logo=docker&logoColor=white"></a>

A flexible and easy tool for serving and scaling PyTorch models in production 🚀⚡🔥<br>


</div>

<br>

### ⚡ Why TorchServe 
<details><summary></summary>
<p>
  
* [Model Management API](docs/management_api.md): multi model management with optimized worker to model allocation
* [Inference API](docs/inference_api.md): REST and gRPC support for batched inference
* [TorchServe Workflows](examples/Workflows/README.md): deploy complex DAGs with multiple interdependent models
* Default way to serve PyTorch models in
  * [Kubeflow](https://v0-5.kubeflow.org/docs/components/pytorchserving/)
  * [MLflow](https://github.com/mlflow/mlflow-torchserve)
  * [Sagemaker](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
  * [Kserve](https://kserve.github.io/website/0.8/modelserving/v1beta1/torchserve/): Supports both v1 and v2 API
  * [Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)
* Export your model for optimized inference. Torchscript out of the box, [ORT](https://discuss.pytorch.org/t/deploying-onnx-model-with-torchserve/97725/2), [IPEX](https://github.com/pytorch/serve/tree/master/examples/intel_extension_for_pytorch), [TensorRT](https://github.com/pytorch/serve/issues/1243), [FasterTransformer](https://github.com/pytorch/serve/tree/master/examples/FasterTransformer_HuggingFace_Bert)
* [Performance Guide](docs/performance_guide.md): builtin support to optimize, benchmark and profile PyTorch and TorchServe performance
* [Expressive handlers](CONTRIBUTING.md): An expressive handler architecture that makes it trivial to support inferencing for your usecase with [many supported out of the box](https://github.com/pytorch/serve/tree/master/ts/torch_handler)
* [Metrics API](docs/metrics.md): out of box support for system level metrics with [Prometheus exports](https://github.com/pytorch/serve/tree/master/examples/custom_metrics), custom metrics and PyTorch profiler support

 </p>
</details>

### 🚀 Quick start with TorchServe
<details><summary></summary>
<p>

#### 1. Clone TorchServe repository
```
git clone https://github.com/pytorch/serve.git
```
#### 2. Install TorchServe and torch-model-archiver
- For CPU

  ```bash
  python ./ts_scripts/install_dependencies.py
  ```

- For GPU with Cuda 10.2. Options are `cu92`, `cu101`, `cu102`, `cu111`, `cu113`

  ```bash
  python ./ts_scripts/install_dependencies.py --cuda=cu113
  ```
#### 3. Store a model
 To serve a model with TorchServe, first archive the model as a MAR file. You can use the model archiver to package a model.
 You can also create model stores to store your archived models.

1. Create a directory to store your models.

    ```bash
    mkdir model_store
    ```

1. Download a trained model.

    ```bash
    wget https://download.pytorch.org/models/densenet161-8d451a50.pth
    ```

1. Archive the model by using the model archiver. The `extra-files` param uses a file from the `TorchServe` repo, so update the path if necessary.

    ```bash
    torch-model-archiver --model-name densenet161 --version 1.0 --model-file ./serve/examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier
    ```
#### 4. Start TorchServe to serve the model

After you archive and store the model, use the `torchserve` command to serve the model.

```bash
torchserve --start --ncs --model-store model_store --models densenet161.mar
```
#### 5. Get predictions from a model
After you execute the `torchserve` command above, TorchServe runs on your host, listening for inference requests.

```bash
curl http://127.0.0.1:8080/predictions/densenet161 -T kitten_small.jpg
```

Which will return the following JSON object

```json
[
  {
    "tiger_cat": 0.46933549642562866
  },
  {
    "tabby": 0.4633878469467163
  },
  {
    "Egyptian_cat": 0.06456148624420166
  },
  {
    "lynx": 0.0012828214094042778
  },
  {
    "plastic_bag": 0.00023323034110944718
  }
]
```
####  6. Stop TorchServe

To stop the currently running TorchServe instance, run:

```bash
torchserve --stop
```

  </p>
  </details>
  
### :crystal_ball: Deploy Custom Model
<details><summary></summary>
<p>

 #### 1. Save weigths from model
 
 Save the model in as an executable script module or a traced script:

1. Save model using scripting
   ```python
   #scripted mode
   import torch
   model = MyModel()
   sm = torch.jit.script(model)
   sm.save("my_model.pt")
   ```

2. Save model using tracing
   ```python
   #traced mode
   import torch
   model = MyModel()
   model.eval()
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("my_model.pt")

 #### 2. Create custom handler
  
 Handler can be TorchServe's inbuilt handler name or path to a py file to handle custom TorchServe inference logic. TorchServe supports the following handlers out of box:
1. `image_classifier`
2. `object_detector`
3. `text_classifier`
4. `image_segmenter`

For a more comprehensive list of built in handlers, make sure to checkout the [examples](https://github.com/pytorch/serve/blob/master/docs/default_handlers.md)
  
Almost you can build your custom handler, for example checkout this [handler] for a plant count model.
 
#### 3. Package your model into .mar file

With the model artifacts available locally, you can use the `torch-model-archiver` CLI to generate a `.mar` file that can be used to serve an inference API with TorchServe.

In this next step we'll run `torch-model-archiver` and tell it our model's name is `plant_count` and its version is `1.0` with the `model-name` and `version` parameter respectively and that it will use a custom handler with the `handler` argument . Then we're giving it the `model-file` and `serialized-file` to the model's assets.

For torchscript:
```bash
torch-model-archiver --model-name plant_count --version 1.0 --serialized-file plantcount.pt --extra-files PlantCountHandler.py --handler my_handler.py --export-path model-store/
```
  
This will package all the model artifacts files and output `plant_count.mar` in the export path working directory. This `.mar` file is all you need to run TorchServe, serving inference requests for a simple image recognition API.

#### 4. Run model with torchserve
  
After you archive and store the model, use the torchserve command to serve the model.
```
torchserve --start --ncs --model-store model_store --models plant_count.mar --no-config-snapshots
```

#### 5. Get prediction
Call the prediction endpoint
```
curl http://127.0.0.1:8080/predictions/plant_count -T ETL-DEEP-XXX.jpg
```

All this steps are for the torchscript mode model, for deploy eager mode model follow this [steps](https://github.com/pytorch/serve/tree/master/examples#creating-mar-file-for-eager-mode-model)

</p>
</details>



