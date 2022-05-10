# Spike-based computational models of bio-inspired memories in the hippocampal CA3 region in SpiNNaker

<h2 name="Description">Description</h2>
<p align="justify">
Code on which the paper entitled "Spike-based computational models of bio-inspired memories in the hippocampal CA3 region in SpiNNaker" is based, accepted and awaiting publication at the congress International Joint Conference on Neural Networks of 2022 <a href="https://wcci2022.org/call-for-papers/">(IJCNN 2022)</a>. 
</p>
<p align="justify">
Two hippocampal bio-inspired memory models implemented on the <a href="https://apt.cs.manchester.ac.uk/projects/SpiNNaker/">SpiNNaker</a> hardware platform using the technology of the Spiking Neuronal Network (SNN) are presented. The code is written in Python and makes use of the PyNN library and their adaptation for SpiNNaker called <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjaxOCWhrn3AhVL1BoKHVtQDvsQFnoECAkQAQ&url=https%3A%2F%2Fgithub.com%2FSpiNNakerManchester%2FsPyNNaker&usg=AOvVaw3e3TBMJ-08yBqtsKza_RiE">sPyNNaker</a>. In addition, the necessary scripts to replicate the tests and plots carried out in the paper are included.
</p>
<p align="justify">
Please go to section <a href="#CiteThisWork">cite this work</a> to learn how to properly reference the works cited here.
</p>

<h2>Table of contents</h2>
<p align="justify">
<ul>
<li><a href="#Description">Description</a></li>
<li><a href="#Article">Article</a></li>
<li><a href="#Instalation">Instalation</a></li>
<li><a href="#Usage">Usage</a></li>
<li><a href="#RepositoryContent">Repository content</a></li>
<li><a href="#CiteThisWork">Cite this work</a></li>
<li><a href="#Credits">Credits</a></li>
<li><a href="#License">License</a></li>
</ul>
</p>


<h2 name="Article">Article</h2>
<p align="justify">
<strong>Title</strong>: Spike-based computational models of bio-inspired memories in the hippocampal CA3 region on SpiNNaker
</p>
<p align="justify">
<strong>Abstract</strong>: The human brain is the most powerful and efficient machine in existence today, surpassing in many ways the capabilities of modern computers. Currently, lines of research in neuromorphic engineering are trying to develop hardware that mimics the functioning of the brain to acquire these superior capabilities. One of the areas still under development is the design of bio-inspired memories, where the hippocampus plays an important role. This region of the brain acts as a short-term memory with the ability to store associations of information from different sensory streams in the brain and recall them later. This is possible thanks to the recurrent collateral network architecture that constitutes CA3, the main sub-region of the hippocampus. In this work, we developed two spike-based computational models of fully functional hippocampal bio-inspired memories for the storage and recall of complex patterns implemented with spiking neural networks on the SpiNNaker hardware platform. These models present different levels of biological abstraction, with the first model having a constant oscillatory activity closer to the biological model, and the second one having an energy-efficient regulated activity, which, although it is still bio-inspired, opts for a more functional approach. Different experiments were performed for each of the models, in order to test their learning/recalling capabilities. A comprehensive comparison between the functionality and the biological plausibility of the presented models was carried out, showing their strengths and weaknesses. The two models, which are publicly available for researchers, could pave the way for future spike-based implementations and applications.
</p>
<p align="justify">
<strong>Keywords</strong>: Hippocampus model, CA3, Neuromorphic engineering, spiking neural networks, SpiNNaker, spike-based memory
</p>
<p align="justify">
<strong>Author</strong>: Daniel Casanueva-Morato
</p>
<p align="justify">
<strong>Contact</strong>: dcasanueva@us.es
</p>


<h2 name="Instalation">Instalation</h2>
<p align="justify">
<ol>
	<li><p align="justify">Have or have access to the SpiNNaker hardware platform. In case of local use, follow the installation instructions available on the <a href="http://spinnakermanchester.github.io/spynnaker/6.0.0/index.html">official website</a></p></li>
	<li><p align="justify">Python version 3.8.10</p></li>
	<li><p align="justify">Python libraries:</p></li>
	<ul>
		<li><p align="justify"><strong>sPyNNaker8</strong>: last stable version <a href="http://spinnakermanchester.github.io/development/gitinstall.html">compiled from source</a></p></li>
		<li><p align="justify"><strong>numpy</strong> 1.21.4</p></li>
		<li><p align="justify"><strong>matplotlib</strong> 3.5.0</p></li>
	</ul>
</ol>
</p>
<p align="justify">
To run any script, follow the python nomenclature: 
<code>
python script.py
</code>
</p>


<h2 name="RepositoryContent">Repository content</h3>
<p align="justify">
<ul>
  <li><p align="justify"><a href="CA3_oscilatory.py">CA3_oscilatory.py</a>: script responsible for building and simulating the oscillating memory model, as well as storing the simulation data in a file.</p></li>
  <li><p align="justify"><a href="CA3_pc_inhibitory.py">CA3_pc_inhibitory.py</a> and <a href="CA3_pc_inhibitory_static_syn.py">CA3_pc_inhibitory_static_syn.py</a>: scripts similar to the above but for the regulated activity model. The former works with the dynamic model (train) and the latter with the static model (test).</p></li>
  <li><p align="justify"><a href="simulation_and_plot_CA3_oscilatory.py">simulation_and_plot_CA3_oscilatory.py</a> and <a href="simulation_and_plot_CA3_pc_inhibitory.py">simulation_and_plot_CA3_pc_inhibitory.py</a>: scripts in charge of carrying out the simulation of the models and the plotting of the necessary graphics on these simulations.</p></li>
  <li><p align="justify"><a href="utils.py">utils.py</a>: set of functions used as tools for the collection, storage, processing and plotting of information from the neuronal network.</p></li>
  <li><p align="justify"><a href="data/">data</a> and <a href="plot/">plot</a>: folders where the data files from the network simulation are stored and where the plots of these data are stored respectively.</p></li>
</ul>
</p>


<h2 name="Usage">Usage</h2>
<p align="justify">
In order to replicate the results of both memory models shown in the paper, it is necessary to select the configuration with which you want to build the memory within the model (comment or uncomment the block of parameters of the experiment you want to replicate) and adjust the time parameter <em>simTime</em> for the duration of the simulation based on this configuration (if the last input of information reaches the model at ms 75, give, for example, a value of 85 ms to this parameter). Once the model has been configured, the <em>simulation_and_plot</em> script corresponding to the model to be tested must be run.
</p>
<p align="justify">
In the case of the regulated activity model (<strong>CA3_pc_inhibitory</strong>), the model must first be trained to learn the patterns, and the result of this training must be passed to the static model to test the recall of the patterns. Therefore, the above steps must first be applied to the dynamic version of the model (<a href="CA3_pc_inhibitory.py">CA3_pc_inhibitory.py</a>) for pattern learning. This will generate a text file in the <a href="data/">data</a> folder with all the simulation information, including the weights of the trained synapses. Next, you have to apply the previous steps again on the static memory model (<a href="CA3_pc_inhibitory_static_syn.py">CA3_pc_inhibitory_static_syn.py</a>), but changing the value of the <em>w_path</em> parameter to the full path to the previously generated file. This way, both phases of the model can be tested separately.
</p>
<p align="justify">
In order to carry out experiments different from the ones performed in the paper, it is enough to modify, instead of selecting, the configuration of the previous experiments to adapt them to the desired experimental conditions. For reference, the main parameters to be modified in the models are: 
</p>
<p align="justify">
<ul>
  <li><p align="justify"><strong>simTime</strong>: indicates how long the simulation will last.</p></li>
  <li><p align="justify"><strong>networkSize</strong>: size of the network in number of neurons. It is directly proportional to the size of the input/output of the network and the size and number of patterns it can store.</p></li>
  <li><p align="justify"><strong>DGLSpikes</strong>: the input spikes to the network. It is a 2d array where it is indicated for each input neuron (first dimension of the array) in which ms it should generate spikes (second dimension of the array).</p></li>
</ul>
</p>


<h2 name="CiteThisWork">Cite this work</h2>
<p align="justify">
Accepted and awaiting publication at the congress International Joint Conference on Neural Networks of 2022 <a href="https://wcci2022.org/call-for-papers/">(IJCNN 2022)</a>. 
</p>


<h2 name="Credits">Credits</h2>
<p align="justify">
The author of the original idea is Daniel Casanueva-Morato while working on a research project of the <a href="http://www.rtc.us.es/">RTC Group</a>.
</p>
<p align="justify">
This research was partially supported by the Spanish grant MINDROB (PID2019-105556GB-C33/AEI/10.13039/501100011033). 
</p>
<p align="justify">
D. C.-M. was supported by a "Formación de Profesor Universitario" Scholarship from the Spanish Ministry of Education, Culture and Sport.
</p>


<h2 name="License">License</h2>
<p align="justify">
This project is licensed under the GPL License - see the <a href="https://github.com/dancasmor/Spike-based-computational-models-of-bio-inspired-memories-in-the-hippocampal-CA3-region-in-SpiNNaker/blob/main/LICENSE">LICENSE.md</a> file for details.
</p>
<p align="justify">
Copyright © 2022 Daniel Casanueva-Morato<br>  
<a href="mailto:dcasanueva@us.es">dcasanueva@us.es</a>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)