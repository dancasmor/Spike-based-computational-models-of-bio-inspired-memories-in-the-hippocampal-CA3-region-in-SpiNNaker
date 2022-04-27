
import spynnaker8 as sim
import utils

"""
Red CA3 simple

+ Poblaciones: 
    + DG: entrada de patrón
    + LEARNING: entrada de activación de la fase de aprendizaje del modelo
    + INH: población inhibitoria de actividad de CA3, ayuda a inhibir aquellas neuronas de CA3 que no 
           deben activarse
    + PC de CA3: la memoria en sí.

+ Sinapsis: 
    + DG-PC: 1 a 1 estáticas excitatoria
    + LEARNING-INH: todas con todas estáticas excitatoria 
    + DG-INH: 1 a 1 estáticas inhibitoria
    + INH-PC: 1 a 1 estáticas inhibitoria
    + PC-PC: todas con todas (salvo consigo mismo) y dinámicas (STDP).
    + PC-PC: todas con todas (salvo consigo mismo) estáticas inhibitorias
"""

# Definición de parámetros:

# + Parámetros de simulación y configuración (ms):
simulationParameters = {"simTime": 95, "timeStep": 1.0, "filename": "CA3_pc_inhibitory"}

# + Número de neuronas por población
networkSize = 15
popNeurons = {"DGLayer": networkSize, "PCLayer": networkSize, "INHLayer": networkSize, "LEARNING": 1}

# + Generación de spikes de entrada en DG y LEARNING

# 2 patrones no ortogonales
DGLSpikes = [[51,61,71,81]]
DGLSpikes = DGLSpikes + [[1,11,21,31, 51,61,71,81] for k in range(12)]
DGLSpikes = DGLSpikes + [[51,61,71,81], [1,11,21,31]]
"""
# 4 patrones ortogonales
DGLSpikes = [[1,11,21,31], [1,11,21,31], [1,11,21,31], [1,11,21,31]]
DGLSpikes = DGLSpikes + [[51,61,71,81], [51,61,71,81], [51,61,71,81], [51,61,71,81]]
DGLSpikes = DGLSpikes + [[101,111,121,131], [101,111,121,131], [101,111,121,131], [101,111,121,131]]
DGLSpikes = DGLSpikes + [[], [], []]
"""
learningStamp = list({x for l in DGLSpikes for x in l})
learningStamp.sort()
LEARNINGSpikes = learningStamp


# + Parámetros de las neuronas
neuronParameters = {
        "PCL": {"cm": 0.27, "i_offset": 0.0, "tau_m": 3.0, "tau_refrac": 2.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
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

# + Parámetros de las sinapsis (peso en uS)
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
    LEARNING = sim.Population(popNeurons["LEARNING"], sim.SpikeSourceArray(spike_times=LEARNINGSpikes), label="LEARNING")
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

    # PC-PC -> conexión stdp todos con todos (salvo consigo misma)
    # + Regla de tiempo -> se usará el tiempo relativo entre spike pre y post
    timing_rule = sim.SpikePairRule(tau_plus=synParameters["PCL-PCL"]["tau_plus"], tau_minus=synParameters["PCL-PCL"]["tau_minus"],
                                    A_plus=synParameters["PCL-PCL"]["A_plus"], A_minus=synParameters["PCL-PCL"]["A_minus"])
    # + Regla de peso -> incremento/decremento de una cantidad sobre el peso actual
    weight_rule = sim.AdditiveWeightDependence(w_max=synParameters["PCL-PCL"]["w_max"], w_min=synParameters["PCL-PCL"]["w_min"])
    # + Modelo de STDP -> definido por las 2 reglas anteriores, el peso inicial de la sinapsis y el delay de aplicación
    stdp_model = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule,
                                   weight=synParameters["PCL-PCL"]["initWeight"], delay=synParameters["PCL-PCL"]["delay"])
    # + Generación de las sinapsis usando la regla stdp modelada anteriormente
    PCL_PCL_conn = sim.Projection(PCLayer, PCLayer, sim.AllToAllConnector(allow_self_connections=False), synapse_type=stdp_model)

    # PCL-PCL -> estáticas inhibitorias todas con todas salvo consigo mismo
    PCL_PCL_inh_conn = sim.Projection(PCLayer, PCLayer, sim.AllToAllConnector(allow_self_connections=False),
                                   synapse_type=sim.StaticSynapse(weight=synParameters["PCL-PCL-inh"]["initWeight"],
                                                                  delay=synParameters["PCL-PCL-inh"]["delay"]),
                                   receptor_type=synParameters["PCL-PCL-inh"]["receptor_type"])

    # LEARNING-INHL -> estática excitatoria todos a todos
    LEARNING_INHL_conn = sim.Projection(LEARNING, INHLayer, sim.AllToAllConnector(allow_self_connections=True),
                                  synapse_type=sim.StaticSynapse(weight=synParameters["LEARNING-INHL"]["initWeight"],
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
    #INHLayer.record(["spikes", "v"])
    #PCLayer.record(["spikes"])

    ######################################
    # Ejecución de la simulación
    ######################################
    # sim.run(simulationParameters["simTime"])
    # La simulación se va a ejecutar en intervalos de tiempo para poder almacenar los valores de los pesos
    w_PCL_PCL = []
    w_PCL_PCL.append(PCL_PCL_conn.get('weight', format='list', with_address=True))  # Instante 0
    for n in range(0, int(simulationParameters["simTime"]), int(simulationParameters["timeStep"])):
        sim.run(simulationParameters["timeStep"])
        w_PCL_PCL.append(PCL_PCL_conn.get('weight', format='list', with_address=True))

    ######################################
    # Tratamiento de datos de salida de la simulación
    ######################################
    # Tomamos los spikes, peso sinapsis y potencial de membrana grabados durante la simulación
    PCData = PCLayer.get_data(variables=["spikes", "v"])
    #INHData = INHLayer.get_data(variables=["spikes", "v"])
    #PCData = PCLayer.get_data(variables=["spikes"])

    # Separamos por tipo de datos -> cada segmento = una simulación/ejecución (run)
    spikesPC = PCData.segments[0].spiketrains
    vPC = PCData.segments[0].filter(name='v')[0]
    #spikesINH = INHData.segments[0].spiketrains
    #vINH = INHData.segments[0].filter(name='v')[0]

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
    #formatVINH = utils.format_neo_data("v", vINH)
    #formatSpikesINH = utils.format_neo_data("spikes", spikesINH)
    formatWeightPCL_PCL = utils.format_neo_data("weights", w_PCL_PCL, {"simTime": simulationParameters["simTime"], "timeStep": simulationParameters["timeStep"]})

    # Muestra de los datos formateados
    # print("V PCLayer = " + str(formatVPC))
    print("Spikes PCLayer = " + str(formatSpikesPC))
    # print("V INHLayer = " + str(formatVINH))
    # print("Spikes INHLayer = " + str(formatSpikesINH))
    # print("Peso sináptico PCL-PCL = " + str(formatWeightPCL_PCL))
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
    """
    dataOut["variables"].append(
        {"type": "spikes", "popName": "INH Layer", "popNameShort": "INHL", "numNeurons": popNeurons["INHLayer"],
         "data": formatSpikesINH})
    dataOut["variables"].append(
        {"type": "v", "popName": "INH Layer", "popNameShort": "INHL", "numNeurons": popNeurons["INHLayer"],
         "data": formatVINH})
    """
    dataOut["variables"].append({"type": "w", "popName": "DGL-PCL", "popNameShort": "PCL-PCL", "data": formatWeightPCL_PCL})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "DG Layer", "popNameShort": "DGL", "numNeurons": popNeurons["DGLayer"],
         "data": DGLSpikes})
    dataOut["variables"].append(
        {"type": "spikes", "popName": "LEARNING Layer", "popNameShort": "LEARNING", "numNeurons": popNeurons["LEARNING"],
         "data": LEARNINGSpikes})

    # Almacenado en fichero
    fullPath, filename = utils.write_file("data/", simulationParameters["filename"], dataOut)
    print("Datos almacenados en: " + fullPath)
    return fullPath, filename


if __name__ == "__main__":
    main()