# Exercise materials for  Applying machine learning methods in research 
## Training for FMI employees, organized together with CSC


## Day 1

MNIST classification with MLPs.

* *01-pytorch-mnist-mlp.ipynb*

Image classification with CNNs.

* *02-pytorch-mnist-cnn.ipynb*

## Day 2 

Land segmentation with UNET.

* *03-train_model.py*
* *03-train_model.sh*
* *03-inference_and_evaluation.ipynb*

## Setup

## Course exercise enviroment
During the course exercises are done in LUMI, which is EuroHPC supercomputer. Accessing LUMI requires LUMI project. Finnish users get access to LUMI via CSC. For course the course participants are added to the course project.

### LUMI web user interface
* Open https://www.lumi.csc.fi
* Log in with:
	* HAKA, if you have (Finnish universities and some research institutes)
 	* [CSC account](https://docs.csc.fi/accounts/), you need your CSC username and password 

### Copy exercise materials
Open Login node shell
```
cd /scratch/project_project_2017263
mkdir $USER
cd $USER
git clone https://github.com/csc-training/lumi-aif-fmi.git
```
	
#### Jupyter 
* Click "Jupyter" on dashboard
* Select following settings:
  	* Reservation: fmi-day1 or fmi-day2
	* Project: project_2017263 during the course, own project later 
	* Partition: interactive
	* CPU cores: 4
	* Local disk: 0
	* Time: 4:00:00 (or adjust to reasonable)
 	* Working directory: /scratch/project_2017263 during the course, own project's scratch later
	* Python: pytorch
		* For exercise 3-4, you need to activate a virtual environment with some extra packages, see below
      		* Check, `Enable virtual environment`
			* Virtual environment path: `/projappl/project_2017263/fmi_course`
   			* Check, `Enable packages under ~/.local/lib on venv start`
   	* (Do not select any of the check-boxes below.)
* Click launch and wait until granted resources 
* Click "Connect to Jupyter"
* Open the cloned exercise folder under your `<your_username>` in JupyterLab


> [!TIP]
> If you see parts of the notebook disappearing when you scroll, this is unfortunately [a known issue with newer versions of JupyterLab](https://github.com/jupyterlab/jupyterlab/issues/17023).
> A workaround is to set the Windowing mode to "defer" as follows:
> - Open "Settings" menu (top bar)
> - Open "Settings Editor"
> - Search for "windowing mode"
> - Set it to "defer", rather than the default "full"


## Acknowledgement

Please acknowledge CSC in your publications, it is important for project continuation and funding reports. As an example, you can write "The authors wish to thank CSC - IT Center for Science, Finland (urn:nbn:fi:research-infras-2016072531) for computational resources and support".
