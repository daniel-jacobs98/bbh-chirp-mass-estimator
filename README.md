<h1>Chirp Mass Estimator for Binary Black Hole GW Signals</h1>
<h2>Daniel Jacobs - OzGrav 2020</h2>
<p>This repo contains code to train models to estimate the chirp mass of binary black holes from their gravitational wave signals. This includes
the estimator model training code, training code for a denoising autoencoder required by the estimator, and code to generate training datasets for the models.</p>

<h2>Dataset Generation</h2>
<p>Code in 'dataset generation' folder.</p>
<p>Includes a python script to generate training datasets of binary black hole signals immersed in Gaussian noise
 coloured by the O2 PSD of the Hanford and Livingston detectors.</p>
<p>Parameter ranges, such as mass ranges, inclination and <b>number of signals to generate</b> are set at the top of the dataset generation file and should be changed there. To generate a dataset on OzGrav, you can batch submit the gen_signals_o2_psd.sh to Slurm. You can change the resource requests and the output file name in that .sh file.</p>
<p>Also includes in the folder are .dat files which contain the PSD of each detector during the O2 science run.</p>
<p>Currently the dataset generation can only produce signals at 2048Hz due to the PSD being used.</p>


<h2>Model Training</h2>
<p>Code in 'model training' folder.</p>
<p>Contains two batch files, which can be submitted to Slurm, and their associated python scripts to train a denoising autoencoder and the chirp mass estimator itself.</p>
<p>The directory which the training data is pulled from can be changed in the python scripts.</p>
<p>Note that as the denoising autoencoder is required for training the chirp mass estimator, the autoencoder must be fully trained before training the estimator.</p>