
import spynnaker8 as sim
import utils

"""
Regulated CA3 network

+ Populations: 
    + DG: pattern input
    + LEARNING: input to activate learning phase
    + INH: inhibitory population to regulate CA3 activity
    + PC of CA3: to store memory

+ Synapses: 
    + DG-PC: 1-to-1, excitatory and static
    + LEARNING-INH: all-to-all, excitatory and static
    + DG-INH: 1-to-1, inhibitory and static
    + INH-PC: 1-to-1, inhibitory and static
    + PC-PC: all-to-all (except with oneself) and dynamic (STDP)
    + PC-PC-inh: all-to-all (except with oneself), inhibitory and static
"""

# Networks parameters
# + Simulation time, time step and base filename of the files generated (ms):
simulationParameters = {"simTime": 95, "timeStep": 1.0, "filename": "CA3_pc_inhibitory"}

# + Size in neurons of the network
networkSize = 15
popNeurons = {"DGLayer": networkSize, "PCLayer": networkSize, "INHLayer": networkSize, "LEARNING": 1}

# + If store the weight or not (large increase in simulation time)
recordWeight = True

# + Input spikes
# 2 non-orthogonal patterns
DGLSpikes = [[51,61,71,81]]
DGLSpikes = DGLSpikes + [[1,11,21,31, 51,61,71,81] for k in range(12)]
DGLSpikes = DGLSpikes + [[51,61,71,81], [1,11,21,31]]
"""
# 4 orthogonal patterns
DGLSpikes = [[1,11,21,31], [1,11,21,31], [1,11,21,31], [1,11,21,31]]
DGLSpikes = DGLSpikes + [[51,61,71,81], [51,61,71,81], [51,61,71,81], [51,61,71,81]]
DGLSpikes = DGLSpikes + [[101,111,121,131], [101,111,121,131], [101,111,121,131], [101,111,121,131]]
DGLSpikes = DGLSpikes + [[], [], []]
"""
learningStamp = list({x for l in DGLSpikes for x in l})
learningStamp.sort()
LEARNINGSpikes = learningStamp


# + Neuron parameters
neuronParameters = {
        "PCL": {"cm": 0.27, "i_offset": 0.0, "tau_m": 3.0, "tau_refrac": 2.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
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

# + Synapses parameters (weight in nA)
synParameters = {
    "DGL-PCL": {"initWeight": 12, "delay": 1.0, "receptor_type": "excitatory"},
    "PCL-PCL": {"tau_plus": 9.0, "tau_minus": 5.0, "A_plus": 6.0, "A_minus": 3.0, "w_max": 12.0, "w_min": 0.0,
               "initWeight": 0.0, "delay": 1.0, "receptor_type": "STDP"},
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
    LEARNING = sim.Population(popNeurons["LEARNING"], sim.SpikeSourceArray(spike_times=LEARNINGSpikes), label="LEARNING")
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

    # PC-PC
    timing_rule = sim.SpikePairRule(tau_plus=synParameters["PCL-PCL"]["tau_plus"], tau_minus=synParameters["PCL-PCL"]["tau_minus"],
                                    A_plus=synParameters["PCL-PCL"]["A_plus"], A_minus=synParameters["PCL-PCL"]["A_minus"])
    weight_rule = sim.AdditiveWeightDependence(w_max=synParameters["PCL-PCL"]["w_max"], w_min=synParameters["PCL-PCL"]["w_min"])
    stdp_model = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule,
                                   weight=synParameters["PCL-PCL"]["initWeight"], delay=synParameters["PCL-PCL"]["delay"])
    PCL_PCL_conn = sim.Projection(PCLayer, PCLayer, sim.AllToAllConnector(allow_self_connections=False), synapse_type=stdp_model)

    # PCL-PCL-inh
    PCL_PCL_inh_conn = sim.Projection(PCLayer, PCLayer, sim.AllToAllConnector(allow_self_connections=False),
                                   synapse_type=sim.StaticSynapse(weight=synParameters["PCL-PCL-inh"]["initWeight"],
                                                                  delay=synParameters["PCL-PCL-inh"]["delay"]),
                                   receptor_type=synParameters["PCL-PCL-inh"]["receptor_type"])

    # LEARNING-INHL
    LEARNING_INHL_conn = sim.Projection(LEARNING, INHLayer, sim.AllToAllConnector(allow_self_connections=True),
                                  synapse_type=sim.StaticSynapse(weight=synParameters["LEARNING-INHL"]["initWeight"],
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
    #INHLayer.record(["spikes", "v"])
    #PCLayer.record(["spikes"])

    ######################################
    # Execute the simulation
    ######################################
    # To store the weight
    if recordWeight:
        w_PCL_PCL = []
        w_PCL_PCL.append(PCL_PCL_conn.get('weight', format='list', with_address=True))  # Instante 0
        for n in range(0, int(simulationParameters["simTime"]), int(simulationParameters["timeStep"])):
            sim.run(simulationParameters["timeStep"])
            w_PCL_PCL.append(PCL_PCL_conn.get('weight', format='list', with_address=True))
    else:
        sim.run(simulationParameters["simTime"])

    ######################################
    # Retrieve output data
    ######################################
    PCData = PCLayer.get_data(variables=["spikes", "v"])
    #INHData = INHLayer.get_data(variables=["spikes", "v"])
    #PCData = PCLayer.get_data(variables=["spikes"])
    spikesPC = PCData.segments[0].spiketrains
    vPC = PCData.segments[0].filter(name='v')[0]
    #spikesINH = INHData.segments[0].spiketrains
    #vINH = INHData.segments[0].filter(name='v')[0]

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
    #formatVINH = utils.format_neo_data("v", vINH)
    #formatSpikesINH = utils.format_neo_data("spikes", spikesINH)
    if recordWeight:
        formatWeightPCL_PCL = utils.format_neo_data("weights", w_PCL_PCL, {"simTime": simulationParameters["simTime"], "timeStep": simulationParameters["timeStep"]})

    # Show some of the data
    # print("V PCLayer = " + str(formatVPC))
    # print("Spikes PCLayer = " + str(formatSpikesPC))
    # print("V INHLayer = " + str(formatVINH))
    # print("Spikes INHLayer = " + str(formatSpikesINH))
    # print("Weight PCL-PCL = " + str(formatWeightPCL_PCL))
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
    """
    dataOut["variables"].append(
        {"type": "spikes", "popName": "INH Layer", "popNameShort": "INHL", "numNeurons": popNeurons["INHLayer"],
         "data": formatSpikesINH})
    dataOut["variables"].append(
        {"type": "v", "popName": "INH Layer", "popNameShort": "INHL", "numNeurons": popNeurons["INHLayer"],
         "data": formatVINH})
    """
    if recordWeight:
        dataOut["variables"].append({"type": "w", "popName": "DGL-PCL", "popNameShort": "PCL-PCL", "data": formatWeightPCL_PCL})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "DG Layer", "popNameShort": "DGL", "numNeurons": popNeurons["DGLayer"],
         "data": DGLSpikes})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "LEARNING Layer", "popNameShort": "LEARNING", "numNeurons": popNeurons["LEARNING"],
         "data": LEARNINGSpikes})

    # Store the data in a file
    fullPath, filename = utils.write_file("data/", simulationParameters["filename"], dataOut)
    print("Data stored in: " + fullPath)
    return fullPath, filename


if __name__ == "__main__":
    main()