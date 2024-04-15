# <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Isabella_Rodriguez.png" alt="Generative Isabella">Attention-based pattern discovery of mass spectrometry imaging data
<p align="center" width="100%">
<img src="Template/pic/workflow.png" alt="Smallville" style="width: 98%; min-width: 300px; display: block; margin: auto;">
</p>
This readme file shows how to properly run the code of Atnal(Jiang et al,):
Guoqing Jiang et al, Attention-based pattern discovery of mass spectrometry imaging data

License:	Attention-based pattern discovery of mass spectrometry imaging data will be shared using the 3D Slicer Software License agreement.

## ABSTRACT
Mass spectrometry imaging (MSI) enables the direct visualization of molecular distributions in tissue sections, making it a crucial
method in metabolomics research. However, the vast size and high dimensionality of MSI data pose challenges for analysis even
though there are already many conventional machine learning methods used in this area, such as the "Curse of dimensionality" problem. Therefore, compressing sparse distributions of raw data while ensuring minimal information loss is important. In response to
these challenges, we propose Atnal, an attention-based generative model. Atnal effectively maps MSI data to a low-dimensional space
with an extremely low loss (2 × 10−7 ∼ 7 × 10−9
), which can contribute to the pattern discovery of MSI data. Then Atnal is applied
in the domain of cancer region recognition and correlation analysis. As it is presented, Atnal can distinguish the regions primarily
containing cancer cells from those with normal cells and identify highly correlated metabolites with cancer (correlation coefficient up
to 0.79). Atnal can provide quantitative guidance for the clinical removal of cancerous tissue, helping to avoid subjective bias and
further aid in clinical cancer diagnosis.

## DEMO
How to run the code?
	1- "Atnal_3DColorectal_trans.ipynb" and "Atnal_3DColorectal_trans.ipynb" are the main files which are independent form each other. We have provided required comments for instructions and guidance. In this file you will be able to:
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

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Wolfgang_Schulz.png" alt="Generative Wolfgang">   Acknowledgements

We encourage you to support the following three amazing artists who have designed the game assets for this project, especially if you are planning to use the assets included here for your own project: 
* Background art: [PixyMoon (@_PixyMoon\_)](https://twitter.com/_PixyMoon_)
* Furniture/interior design: [LimeZu (@lime_px)](https://twitter.com/lime_px)
* Character design: [ぴぽ (@pipohi)](https://twitter.com/pipohi)

In addition, we thank Lindsay Popowski, Philip Guo, Michael Terry, and the Center for Advanced Study in the Behavioral Sciences (CASBS) community for their insights, discussions, and support. Lastly, all locations featured in Smallville are inspired by real-world locations that Joon has frequented as an undergraduate and graduate student---he thanks everyone there for feeding and supporting him all these years.