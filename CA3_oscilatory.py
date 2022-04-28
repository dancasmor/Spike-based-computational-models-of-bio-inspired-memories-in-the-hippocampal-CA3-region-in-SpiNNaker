
import spynnaker8 as sim
import utils

"""
Oscilatory CA3 network

+ Populations: 
    + DG: pattern input
    + PC of CA3: to store memory  

+ Synapses: 
    + DG-PC: 1-to-1, excitatory and static
    + PC-PC: all-to-all (except with oneself) and dynamic (STDP)
    + PC-PC: all-to-all (except with oneself), inhibitory and static
"""

# Networks parameters
# + Simulation time, time step and base filename of the files generated (ms):
simulationParameters = {"simTime": 85, "timeStep": 1.0, "filename": "CA3_oscilatory"}

# + Size in neurons of the network
networkSize = 15
popNeurons = {"DGLayer": networkSize, "PCLayer": networkSize}

# + If store the weight or not (large increase in simulation time)
recordWeight = False

# + Input spikes
# 3 orthogonal patterns
DGLSpikes = [[1,2,3,4,5]]
DGLSpikes = DGLSpikes + [[1,2,3,4,5, 43,44,45,46,47] for i in range(4)]
DGLSpikes = DGLSpikes + [[29,30,31,32,33, 71,72,73,74,75] for j in range(4)]
DGLSpikes = DGLSpikes + [[29,30,31,32,33]]
DGLSpikes = DGLSpikes + [[15,16,17,18,19, 57,58,59,60,61] for k in range(4)]
DGLSpikes = DGLSpikes + [[15,16,17,18,19]]
"""
# 2 orthogonal patterns
DGLSpikes = [[1,2,3,4,5]]
DGLSpikes = DGLSpikes + [[1,2,3,4,5, 41,42,43,43,45] for i in range(4)]
DGLSpikes = DGLSpikes + [[] for j in range(5)]
DGLSpikes = DGLSpikes + [[21,22,23,24,25, 61,62,63,64,65] for k in range(4)]
DGLSpikes = DGLSpikes + [[21,22,23,24,25]]
"""
"""
# 2 non-orthogonal patterns
DGLSpikes = [[1,2,3,4,5]]
DGLSpikes = DGLSpikes + [[1,2,3,4,5, 41,42,43,44,45] for i in range(4)]
DGLSpikes = DGLSpikes + [[] for j in range(2)]
DGLSpikes = DGLSpikes + [[1,2,3,4,5, 21,22,23,24,25]]
DGLSpikes = DGLSpikes + [[] for l in range(2)]
DGLSpikes = DGLSpikes + [[21,22,23,24,25, 61,62,63,64,65] for k in range(4)]
DGLSpikes = DGLSpikes + [[21,22,23,24,25]]
"""

# + Neuron parameters
neuronParameters = {
        "PCL": {"cm": 0.27, "i_offset": 0.0, "tau_m": 3.0, "tau_refrac": 1.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                "v_reset": -60.0, "v_rest": -60.0, "v_thresh": -55.0},
        "DGL": "Source Spike"
}

# + Neuron initial parameters
initNeuronParameters = {
    "PCL": {"vInit": -60},
    "DGL": {"vInit": False}
}

# + Synapses parameters (weight in nA)
synParameters = {
    "DGL-PCL": {"initWeight": 6.0*(popNeurons["PCLayer"]-1), "delay": 1.0, "receptor_type": "excitatory"},
    "PCL-PCL": {"tau_plus": 3.0, "tau_minus": 2.0, "A_plus": 6.0, "A_minus": 3.0, "w_max": 12.0, "w_min": 0.0,
               "initWeight": 0.0, "delay": 1.0, "receptor_type": "STDP"},
    "PCL-PCL-inh": {"initWeight": 1.5, "delay": 1.0, "receptor_type": "inhibitory"}
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

    ######################################
    # Parameters to store
    ######################################
    PCLayer.record(["spikes", "v"])

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
    spikesPC = PCData.segments[0].spiketrains
    vPC = PCData.segments[0].filter(name='v')[0]

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
    if recordWeight:
        formatWeightPCL_PCL = utils.format_neo_data("weights", w_PCL_PCL, {"simTime": simulationParameters["simTime"], "timeStep": simulationParameters["timeStep"]})

    # Show some of the data
    # print("V PCLayer = " + str(formatVPC))
    # print("Spikes PCLayer = " + str(formatSpikesPC))
    # print("Weight PCL-PCL = " + str(formatWeightPCL_PCL))
    # print("Spikes DGL = " + str(DGLSpikes))

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
    if recordWeight:
        dataOut["variables"].append({"type": "w", "popName": "DGL-PCL", "popNameShort": "PCL-PCL", "data": formatWeightPCL_PCL})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "DG Layer", "popNameShort": "DGL", "numNeurons": popNeurons["DGLayer"],
         "data": DGLSpikes})

    # Store the data in a file
    fullPath, filename = utils.write_file("data/", simulationParameters["filename"], dataOut)
    print("Data stored in: " + fullPath)
    return fullPath, filename


if __name__ == "__main__":
    main()