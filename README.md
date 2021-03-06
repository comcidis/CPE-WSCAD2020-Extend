# CPE-WSCAD2020-Extend
 Repository for CPE - WSCAD2020-Extend
 
> Attention, when replicating the experiment, pay attention to the location of the database according to the layout of your machine, you can change it directly in the .py code.

### Here are the versions of all the tools, frameworks and environment.
    Python: 3.7.5
    Pandas: 1.0.4
    Numpy: 1.18.4
    Sckit-learn: 0.23.1
    Ubuntu: 18.04.5 LTS
    kernel: 5.4.0-66-generic
   
### Below is the distribution of the folders of the main experiments.

	|Phase 1
	    |arvore
		|arvore_class
		|arvore_reg
	    |floresta
		|floresta_class
		|floresta_reg

	|Phase 2
	    |arvore
		|arvore_class
		|arvore_reg
	    |floresta
		|floresta_class
		|floresta_reg	 	
	    |xGboost
		|cpu
		|gpu 	

### Folder information:


### Phase 1

Phase 1 Uses the databases in the "BaseSintetica" folder and the "BaseNova" folder. The data sets vary between 25 thousand, 100 thousand, 500 thousand, 2 million, 3 million and 5 million examples. The configuration of the features varies between 5, 10 and 21, below this distribution of the features is shown in more detail.

	5 features: 2 numeric and 3 categorical

	10 features: 5 numeric and 5 categorical

	21 features: 10 numeric and 11 categorical



### Phase 2

Phase 2 is more complex, as we will be varying the parameters, which generate large results. In addition, we are using real data, and measuring accuracy, so a database has been separated for classification ("HIGGS dataset") and for regression ("Seoul Bike Trip Duration Prediction dataset").


For classification, we have the following definition.

	Dataset: HIGGS defined in 5 million examples and 28 attributes.

	Modified parameters:
	max_depth = (default = 66,8,16,33)
	min_sample_split = (default = 2,4,8,16)
	* max_depth = default = 66 (when the tree is built without defining a value, the value goes to 66.)
	

	
For regression, we have the following definition.

	Dataset: Seoul Bike Trip Duration Prediction in 5 million examples and 24 attributes.

	Modified parameters:
	max_depth = (default = 73,8,16,33,66)
	min_sample_split = (default, 4,8,16)
	* max_depth = default = 73 (when the tree is built without defining a value, the value goes to 73.)
	





Important links

All datasets are available in (https://zenodo.org/record/4723678#.YJvesKhKiUl).


For the hotspots evaluation, Random Forest and XGBoost algorithms  were implemented by Erik Linder-Nor??(https://github.com/eriklindernoren/ML-From-Scratch), while the CART Classifier and CART Regression  algorithms  were implemented by Zaur Fataliyev(https://github.com/zziz/cart).
