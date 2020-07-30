# Computer Pointer Controller

<!--*TODO:* Write a short introduction to your project-->

The project is a demonstration of a inference pipeline built using the OpenVino toolkit combining multiple models to enable you to control your mouse pointer position with your gaze. This involves the use of four separate neural network models and various other image processing to accomplish. An overview of the models and pipeline is below 

![Processing Pipeline](pipeline.png)

## Project Set Up and Installation
<!-- *TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.-->
This project was developed using the following prerequisites: 
>  - Intel processor supported by Openvino  
>  - Ubuntu 18 
>  - git 
>  - Python => 3.6
>  - Openvino 2020.1 with GPU support installed

After meeting the prerequisites, you can continue set up as follows:

- Clone project repo  
`git clone https://github.com/ltstein/Udacity-Edge-AI-MouseControllerProject.git`

- Download models using base openvino install (Tested using 2020.1) Run from within the project directory (default is Udacity-Edge-AI-MouseControllerProject/)  
`python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --list models.lst -o models/`

- Install pyautogui dependency  
`sudo apt-get install python3-tk python3-dev`  

- Set up virtual env  
`python3 -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  

## Demo
<!--*TODO:* Explain how to run a basic demo of your model.-->
Enter the project directory and activate the virtual environment. Ensure your openvino install is initialized. Then, a simple demo can be run with just the following:  

`python src/main.py'

This will run the pipeline on a single demo image, allowing you to view some intermediate processing steps and model outputs drawn on the images. This also disables the mouse controller portion of the code.

## Documentation
<!--*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.-->

The following command line arguments are supported with the options and defaults listed:

    --device, default='CPU' #CPU, GPU  
        Sets device for inference
    --hpe, default='FP32') #FP16, FP32, FP32-INT  
        Sets Head Pose Estimation model precision
    --fld, default='FP32') #FP16, FP32, FP32-INT8  
        Sets Facial Landmark Detector model precision
    --ge, default='FP32')  #FP16, FP32, FP32-INT8  
        Sets Gaze Estimation model precision
    --input_type, default='image') #video, cam, image  
        Sets input type
    --input_file, default='bin/demo_1.png') #path to file  
        Sets input file path
    --v default = False  
        Sets verbose output to console

The following example commands are valid  
`python src/main.py`  
`python src/main.py --device 'GPU'`  
`python src/main.py --device 'GPU' --hpe 'FP16' --fld 'FP16' --ge 'FP16'`  
`python src/main.py --input_type 'image' --input_file 'bin/demo_1.png' --v`  

## Benchmarks
<!--*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.-->

Below are example benchmark results obtained using an  i7-7700HQ CPU

| Name                        | Load FP32 | Load FP16 | Load % Diff | Inference FP32 | Inference FP16 | Inference % Diff |
|-----------------------------|-----------|-----------|-------------|----------------|----------------|------------------|
| Face Detection              | 0.171     | -         | -           | 0.0114         | -              | -                |
| Facial Landmarks  Detection | 0.064     | 0.073     | +14         | 0.0006         | 0.0005         | -16              |
| Head Pose Estimation        | 0.049     | 0.047     | -4          | 0.0017         | 0.0012         | -29              |
| Gaze Estimation             | 0.077     | 0.085     | +10         | 0.0021         | 0.0015         | -28              |

| Name                        | Load FP32 | Load FP16 | Load % Diff | Inference FP32 | Inference FP16 | Inference % Diff |
|-----------------------------|-----------|-----------|-------------|----------------|----------------|------------------|
| Face Detection              | 23.44     | -         | -           | 0.0173         | -              | -                |
| Facial Landmarks  Detection | 4.942     | 5.184     | +5          | 0.0012         | 0.0014         | +16              |
| Head Pose Estimation        | 3.435     | 3.332     | -3          | 0.0022         | 0.0020         | -9               |
| Gaze Estimation             | 5.376     | 5.498     | +2          | 0.0026         | 0.0038         | +46              |


## Results
<!--*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.-->
The load times are relatively similar for the CPU, however the inference times show a relatively significant increase in performance when used in FP16. Comparing this to the GPU, there is a large increase in model load times and a much different effect switching to FP16 models. Further investigation is needed to optimze the performance of the models using the DL Workbench and the application using Vtune.
<!--
## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.
-->
### Edge Cases
<!--There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.-->
This application has several edge cases that can cause errors or crashes in the pipeline. A common one is data needed for the next model is not available, such as a missing eye from the facial landmarks due to the users pose. Another is not detecting any face initially. These and several other edge cases are handled by checking model outputs throughout the application and skipping frames where data is missing. Currently, there is little post processing done before the cursor control is used from the gaze estimation, so moderate control parameters were implemented to try to mitigate the effect of wildy changing gaze vectors. 
