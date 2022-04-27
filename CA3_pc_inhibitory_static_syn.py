
# import pyNN.spiNNaker as sim
import spynnaker8 as sim
import utils

"""
Red CA3 inhibitoria con sinapsis estáticas entre PC-PC. Estos pesos serán tomados del resultado del aprendizaje anterior

"""

# Definición de parámetros:

# + Parámetros de simulación y configuración (ms):
simulationParameters = {"simTime": 55, "timeStep": 1.0, "filename": "CA3_pc_inhibitory_static"}

# + Número de neuronas por población
networkSize = 15
popNeurons = {"DGLayer": networkSize, "PCLayer": networkSize, "INHLayer": networkSize, "LEARNING": 1}

# + Generación de spikes de entrada en DG

# 2 patrones no ortogonales
DGLSpikes = [[1,4]]
DGLSpikes = DGLSpikes + [[1,4, 15,18, 30,33, 45,48] for i in range(8)]
DGLSpikes = DGLSpikes + [[] for j in range(6)]

"""
# 4 patrones ortogonales
DGLSpikes = [[1], [1], [1], []]
DGLSpikes = DGLSpikes + [[11], [11], [11], []]
DGLSpikes = DGLSpikes + [[21], [21], [21], []]
DGLSpikes = DGLSpikes + [[], [], []]
"""

LEARNINGSpikes = []

# + Parámetros de las neuronas
neuronParameters = {
        "PCL": {"cm": 0.27, "i_offset": 0.0, "tau_m": 5.0, "tau_refrac": 2.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                "v_reset": -60.0, "v_rest": -60.0, "v_thresh": -55.0},
        "DGL": "Source Spike",
        "INHL": {"cm": 0.27, "i_offset": 0.0, "tau_m": 3.0, "tau_refrac": 2.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                "v_reset": -60.0, "v_rest": -60.0, "v_thresh": -55.0},
        "LEARNING": "Source Spike"
}

# + Parámetros iniciales de las neuronas
initNeuronParameters = {
    "PCL": {"vInit": -60},
    "DGL": {"vInit": False},
    "INHL": {"vInit": -60},
    "LEARNING": {"vInit": False}
}

# + Parámetros de las sinapsis (peso en nA)
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
    # Establecimiento de parámetros de la simulación y spinnaker
    ######################################
    # El paso de tiempo de cada iteración en ms
    sim.setup(timestep=simulationParameters["timeStep"])

    ######################################
    # Creación de poblaciones
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
    # Creación de sinapsis
    ######################################

    # DG-PC -> estática 1 a 1
    DGL_PCL_conn = sim.Projection(DGLayer, PCLayer, sim.OneToOneConnector(),
                                  synapse_type=sim.StaticSynapse(weight=synParameters["DGL-PCL"]["initWeight"],
                                                                 delay=synParameters["DGL-PCL"]["delay"]),
                                  receptor_type=synParameters["DGL-PCL"]["receptor_type"])

    # PC-PC -> conexión estática todos con todos (salvo consigo misma)
    # + Abrimos el archivo y elaboramos la lista de conexiones con los pesos de la última iteración
    synapsePCL_PCL, synParametersOrigin = utils.get_last_stamp_synapse_list(synParameters["PCL-PCL-origin"]["initWeight"])

    PCL_PCL_conn = sim.Projection(PCLayer, PCLayer, sim.FromListConnector(synapsePCL_PCL),
                                  synapse_type=sim.StaticSynapse(), receptor_type="excitatory")

    # + Añadimos a los metaparámetros de las sinapsis el original
    synParameters["PCL-PCL"] = synParametersOrigin

    # PCL-PCL -> estáticas inhibitorias todas con todas salvo consigo mismo
    PCL_PCL_inh_conn = sim.Projection(PCLayer, PCLayer, sim.AllToAllConnector(allow_self_connections=False),
                                      synapse_type=sim.StaticSynapse(weight=synParameters["PCL-PCL-inh"]["initWeight"],
                                                                     delay=synParameters["PCL-PCL-inh"]["delay"]),
                                      receptor_type=synParameters["PCL-PCL-inh"]["receptor_type"])

    # LEARNING-INHL -> estática excitatoria todos a todos
    LEARNING_INHL_conn = sim.Projection(LEARNING, INHLayer, sim.AllToAllConnector(allow_self_connections=True),
                                        synapse_type=sim.StaticSynapse(
                                            weight=synParameters["LEARNING-INHL"]["initWeight"],
                                            delay=synParameters["LEARNING-INHL"]["delay"]),
                                        receptor_type=synParameters["LEARNING-INHL"]["receptor_type"])

    # DGL-INHL -> estáticas inhibitorias 1 a 1
    DGL_INHL_conn = sim.Projection(DGLayer, INHLayer, sim.OneToOneConnector(),
                                   synapse_type=sim.StaticSynapse(weight=synParameters["DGL-INHL"]["initWeight"],
                                                                  delay=synParameters["DGL-INHL"]["delay"]),
                                   receptor_type=synParameters["DGL-INHL"]["receptor_type"])

    # INHL-PCL -> estáticas inhibitorias 1 a 1
    INHL_PCL_conn = sim.Projection(INHLayer, PCLayer, sim.OneToOneConnector(),
                                   synapse_type=sim.StaticSynapse(weight=synParameters["INHL-PCL"]["initWeight"],
                                                                  delay=synParameters["INHL-PCL"]["delay"]),
                                   receptor_type=synParameters["INHL-PCL"]["receptor_type"])

    ######################################
    # Establecimiento de parámetros a grabar
    ######################################
    # Tomamos los spikes y potencial de membrana de PC e INH
    PCLayer.record(["spikes", "v"])
    INHLayer.record(["spikes", "v"])

    ######################################
    # Ejecución de la simulación
    ######################################
    sim.run(simulationParameters["simTime"])
    # La simulación se va a ejecutar en intervalos de tiempo para poder almacenar los valores de los pesos
    """
    w_PCL_PCL = []
    w_PCL_PCL.append(PCL_PCL_conn.get('weight', format='list', with_address=True))  # Instante 0
    for n in range(0, int(simulationParameters["simTime"]), int(simulationParameters["timeStep"])):
        sim.run(simulationParameters["timeStep"])
        w_PCL_PCL.append(PCL_PCL_conn.get('weight', format='list', with_address=True))
    """
    ######################################
    # Tratamiento de datos de salida de la simulación
    ######################################
    # Tomamos los spikes, peso sinapsis y potencial de membrana grabados durante la simulación
    PCData = PCLayer.get_data(variables=["spikes", "v"])
    INHData = INHLayer.get_data(variables=["spikes", "v"])

    # Separamos por tipo de datos -> cada segmento = una simulación/ejecución (run)
    spikesPC = PCData.segments[0].spiketrains
    vPC = PCData.segments[0].filter(name='v')[0]
    spikesINH = INHData.segments[0].spiketrains
    vINH = INHData.segments[0].filter(name='v')[0]

    ######################################
    # Finalización de la simulación
    ######################################
    sim.end()

    ######################################
    # Almacenamiento de la información
    ######################################
    # Formateo de la información de salida de la red
    formatVPC = utils.format_neo_data("v", vPC)
    formatSpikesPC = utils.format_neo_data("spikes", spikesPC)
    formatVINH = utils.format_neo_data("v", vINH)
    formatSpikesINH = utils.format_neo_data("spikes", spikesINH)
    """
    formatWeightPCL_PCL = utils.format_neo_data("weights", w_PCL_PCL, {"simTime": simulationParameters["simTime"],
                                                                       "timeStep": simulationParameters["timeStep"]})
    """

    # Muestra de los datos formateados
    #print("V PCLayer = " + str(formatVPC))
    print("Spikes PCLayer = " + str(formatSpikesPC))
    #print("V INHLayer = " + str(formatVINH))
    #print("Spikes INHLayer = " + str(formatSpikesINH))
    #print("Peso sináptico PCL-PCL = " + str(formatWeightPCL_PCL))
    print("Spikes DGL = " + str(DGLSpikes))
    print("Spikes LEARNING = " + str(LEARNINGSpikes))

    # Organizando el formato del fichero y añadiendo cabeceras
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
    """
    dataOut["variables"].append(
        {"type": "w", "popName": "DGL-PCL", "popNameShort": "PCL-PCL", "data": formatWeightPCL_PCL})
    """
    dataOut["variables"].append(
        {"type": "spikes", "popName": "DG Layer", "popNameShort": "DGL", "numNeurons": popNeurons["DGLayer"],
         "data": DGLSpikes})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "LEARNING Layer", "popNameShort": "LEARNING",
         "numNeurons": popNeurons["LEARNING"],
         "data": LEARNINGSpikes})

    # Almacenado en fichero
    fullPath, filename = utils.write_file("data/", simulationParameters["filename"], dataOut)
    print("Datos almacenados en: " + fullPath)
    return fullPath, filename


if __name__ == "__main__":
    main()
