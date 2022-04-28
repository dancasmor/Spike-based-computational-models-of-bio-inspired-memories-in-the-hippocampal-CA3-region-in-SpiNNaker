
# import pyNN.spiNNaker as sim
import spynnaker8 as sim
import utils

"""
Regulated CA3 static network (weights from dinamic network)
"""

# Networks parameters
# + Simulation time, time step and base filename of the files generated (ms):
simulationParameters = {"simTime": 55, "timeStep": 1.0, "filename": "CA3_pc_inhibitory_static"}

# + Size in neurons of the network
networkSize = 15
popNeurons = {"DGLayer": networkSize, "PCLayer": networkSize, "INHLayer": networkSize, "LEARNING": 1}

# + Input spikes
# 2 non-orthogonal patterns
DGLSpikes = [[1,4]]
DGLSpikes = DGLSpikes + [[1,4, 15,18, 30,33, 45,48] for i in range(8)]
DGLSpikes = DGLSpikes + [[] for j in range(6)]
"""
# 4 orthogonal patterns
DGLSpikes = [[1], [1], [1], []]
DGLSpikes = DGLSpikes + [[11], [11], [11], []]
DGLSpikes = DGLSpikes + [[21], [21], [21], []]
DGLSpikes = DGLSpikes + [[], [], []]
"""

LEARNINGSpikes = []

# + Neuron parameters
neuronParameters = {
        "PCL": {"cm": 0.27, "i_offset": 0.0, "tau_m": 5.0, "tau_refrac": 2.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                "v_reset": -60.0, "v_rest": -60.0, "v_thresh": -55.0},
        "DGL": "Source Spike",
        "INHL": {"cm": 0.27, "i_offset": 0.0, "tau_m": 3.0, "tau_refrac": 2.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                "v_reset": -60.0, "v_rest": -60.0, "v_thresh": -55.0},
        "LEARNING": "Source Spike"
}

# + Neuron initial parameters
initNeuronParameters = {
    "PCL": {"vInit": -60},
    "DGL": {"vInit": False},
    "INHL": {"vInit": -60},
    "LEARNING": {"vInit": False}
}

# + Synapses parameters (weight in nA): path to the data file with trained network
w_path = "data/CA3_pc_inhibitory_2022_01_25__11_38_12.txt"
synParameters = {
    "DGL-PCL": {"initWeight": 12.0, "delay": 1.0, "receptor_type": "excitatory"},
    "PCL-PCL": {},
    "PCL-PCL-origin": {"initWeight": w_path, "delay": 1.0, "receptor_type": "excitatory"},
    "LEARNING-INHL": {"initWeight": 8.0, "delay": 1.0, "receptor_type": "excitatory"},
    "DGL-INHL": {"initWeight": 8.0, "delay": 1.0, "receptor_type": "inhibitory"},
    "INHL-PCL": {"initWeight": 12.0*networkSize, "delay": 1.0, "receptor_type": "inhibitory"},
    "PCL-PCL-inh": {"initWeight": 11.0, "delay": 1.0, "receptor_type": "inhibitory"}
}


def main():

    ######################################
    # Simulation parameters
    ######################################
    sim.setup(timestep=simulationParameters["timeStep"])

    ######################################
    # Create neuron population
    ######################################
    # DG
    DGLayer = sim.Population(popNeurons["DGLayer"], sim.SpikeSourceArray(spike_times=DGLSpikes), label="DGLayer")
    # PC
    PCLayer = sim.Population(popNeurons["PCLayer"], sim.IF_curr_exp(**neuronParameters["PCL"]), label="PCLayer")
    PCLayer.set(v=initNeuronParameters["PCL"]["vInit"])
    # LEARNING
    LEARNING = sim.Population(popNeurons["LEARNING"], sim.SpikeSourceArray(spike_times=LEARNINGSpikes),
                              label="LEARNING")
    # INH
    INHLayer = sim.Population(popNeurons["INHLayer"], sim.IF_curr_exp(**neuronParameters["INHL"]), label="INHLayer")
    PCLayer.set(v=initNeuronParameters["INHL"]["vInit"])

    ######################################
    # Create synapses
    ######################################

    # DG-PC
    DGL_PCL_conn = sim.Projection(DGLayer, PCLayer, sim.OneToOneConnector(),
                                  synapse_type=sim.StaticSynapse(weight=synParameters["DGL-PCL"]["initWeight"],
                                                                 delay=synParameters["DGL-PCL"]["delay"]),
                                  receptor_type=synParameters["DGL-PCL"]["receptor_type"])

    # PC-PC: statics
    # + Take weight from last iteration
    synapsePCL_PCL, synParametersOrigin = utils.get_last_stamp_synapse_list(synParameters["PCL-PCL-origin"]["initWeight"])

    PCL_PCL_conn = sim.Projection(PCLayer, PCLayer, sim.FromListConnector(synapsePCL_PCL),
                                  synapse_type=sim.StaticSynapse(), receptor_type="excitatory")

    # + Assign to synapses
    synParameters["PCL-PCL"] = synParametersOrigin

    # PCL-PCL-inh
    PCL_PCL_inh_conn = sim.Projection(PCLayer, PCLayer, sim.AllToAllConnector(allow_self_connections=False),
                                      synapse_type=sim.StaticSynapse(weight=synParameters["PCL-PCL-inh"]["initWeight"],
                                                                     delay=synParameters["PCL-PCL-inh"]["delay"]),
                                      receptor_type=synParameters["PCL-PCL-inh"]["receptor_type"])

    # LEARNING-INHL
    LEARNING_INHL_conn = sim.Projection(LEARNING, INHLayer, sim.AllToAllConnector(allow_self_connections=True),
                                        synapse_type=sim.StaticSynapse(
                                            weight=synParameters["LEARNING-INHL"]["initWeight"],
                                            delay=synParameters["LEARNING-INHL"]["delay"]),
                                        receptor_type=synParameters["LEARNING-INHL"]["receptor_type"])

    # DGL-INHL
    DGL_INHL_conn = sim.Projection(DGLayer, INHLayer, sim.OneToOneConnector(),
                                   synapse_type=sim.StaticSynapse(weight=synParameters["DGL-INHL"]["initWeight"],
                                                                  delay=synParameters["DGL-INHL"]["delay"]),
                                   receptor_type=synParameters["DGL-INHL"]["receptor_type"])

    # INHL-PCL
    INHL_PCL_conn = sim.Projection(INHLayer, PCLayer, sim.OneToOneConnector(),
                                   synapse_type=sim.StaticSynapse(weight=synParameters["INHL-PCL"]["initWeight"],
                                                                  delay=synParameters["INHL-PCL"]["delay"]),
                                   receptor_type=synParameters["INHL-PCL"]["receptor_type"])

    ######################################
    # Parameters to store
    ######################################
    PCLayer.record(["spikes", "v"])
    INHLayer.record(["spikes", "v"])

    ######################################
    # Execute the simulation
    ######################################
    sim.run(simulationParameters["simTime"])
    
    ######################################
    # Retrieve output data
    ######################################
    PCData = PCLayer.get_data(variables=["spikes", "v"])
    INHData = INHLayer.get_data(variables=["spikes", "v"])
    spikesPC = PCData.segments[0].spiketrains
    vPC = PCData.segments[0].filter(name='v')[0]
    spikesINH = INHData.segments[0].spiketrains
    vINH = INHData.segments[0].filter(name='v')[0]

    ######################################
    # End simulation
    ######################################
    sim.end()

    ######################################
    # Processing and store the output data
    ######################################
    # Format the retrieve data
    formatVPC = utils.format_neo_data("v", vPC)
    formatSpikesPC = utils.format_neo_data("spikes", spikesPC)
    formatVINH = utils.format_neo_data("v", vINH)
    formatSpikesINH = utils.format_neo_data("spikes", spikesINH)

    # Show some of the data
    # print("V PCLayer = " + str(formatVPC))
    # print("Spikes PCLayer = " + str(formatSpikesPC))
    # print("V INHLayer = " + str(formatVINH))
    # print("Spikes INHLayer = " + str(formatSpikesINH))
    # print("Spikes DGL = " + str(DGLSpikes))
    # print("Spikes LEARNING = " + str(LEARNINGSpikes))

    # Create a dictionary with all the information and headers
    dataOut = {"scriptName": simulationParameters["filename"], "timeStep": simulationParameters["timeStep"],
               "simTime": simulationParameters["simTime"], "synParameters": synParameters,
               "neuronParameters": neuronParameters, "initNeuronParameters": initNeuronParameters, "variables": []}
    dataOut["variables"].append(
        {"type": "spikes", "popName": "PC Layer", "popNameShort": "PCL", "numNeurons": popNeurons["PCLayer"],
         "data": formatSpikesPC})
    dataOut["variables"].append(
        {"type": "v", "popName": "PC Layer", "popNameShort": "PCL", "numNeurons": popNeurons["PCLayer"],
         "data": formatVPC})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "INH Layer", "popNameShort": "INHL", "numNeurons": popNeurons["INHLayer"],
         "data": formatSpikesINH})
    dataOut["variables"].append(
        {"type": "v", "popName": "INH Layer", "popNameShort": "INHL", "numNeurons": popNeurons["INHLayer"],
         "data": formatVINH})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "DG Layer", "popNameShort": "DGL", "numNeurons": popNeurons["DGLayer"],
         "data": DGLSpikes})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "LEARNING Layer", "popNameShort": "LEARNING",
         "numNeurons": popNeurons["LEARNING"],
         "data": LEARNINGSpikes})

    # Store the data in a file
    fullPath, filename = utils.write_file("data/", simulationParameters["filename"], dataOut)
    print("Datos almacenados en: " + fullPath)
    return fullPath, filename


if __name__ == "__main__":
    main()
