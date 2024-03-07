This readme file shows how to properly run the code of Atnal(Jiang et al,):
Guoqing Jiang et al, Attention-based pattern discovery of mass spectrometry imaging data

License:	Attention-based pattern discovery of mass spectrometry imaging data will be shared using the 3D Slicer Software License agreement.


---------------------------------- Demo --------------------------------------
How to run the code?
	1- "msiPL_Run_3DColorectal_trans.ipynb" and "msiPL_Run_3DColorectal_trans.ipynb" are the main files which are independent form each other. We have provided required comments for instructions and guidance. In this file you will be able to:
		1.1. Load a dataset.
		1.2. Load the computational neural network architecture (Atnal).
		1.3. Train the model.
		1.3. non-linear dimensionality reduction
		1.4. Evaluate the learning quality by estimation and reconstruction of the original data
		1.6. Perform data clustering (GMM).
		1.7. Identify localized peaks within each cluster.
		
	2- "Computational_Model_trans.py" and "gmm.py": implementation of the Atnal(our neural networks).
	
	3- "LearnPeaks.py": implementation of a function that identifies peaks of interest. 
		It should be called after training the model.

------------------------------------------------------------------------------------
We provide a sample of a publicly available MSI data to train and test the model and to ensure reproducibility.