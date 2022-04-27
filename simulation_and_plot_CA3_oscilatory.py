
import random
import utils
import CA3_oscilatory


def custom_plots(fullPathFile, plot, save, saveName, savePath):
    """
    Función de prueba de apertura de archivo y generación de los plots personalizados

    :param fullPathFile: path + filename al archivo con los datos de la simulación
    :param plot: bool que indica si se quiere o no mostrar los plots generados
    :param save: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return: True si se ha ejecutado correctamente o False si ha habido problemas
    """
    # Abrimos el archivo
    data = utils.read_file(fullPathFile)
    if not data:
        print("Error al abrir el fichero de datos")
        return False
    # Creamos la carpeta donde almacenar los plots
    savePath = utils.check_folder(savePath + saveName + "/")
    if not savePath:
        print("Error al crear la carpeta de almacenamiento")
        return False

    # Buscamos las variables v y spikes
    vPC, spikesPC, spikesDG, wPC_PC, vINH, spikesINH = {}, {}, {}, {}, {}, {}
    for variable in data["variables"]:
        if variable["type"] == "v" and variable["popNameShort"] == "PCL":
            vPC = variable
        elif variable["type"] == "spikes" and variable["popNameShort"] == "PCL":
            spikesPC = variable
        elif variable["type"] == "spikes" and variable["popNameShort"] == "DGL":
            spikesDG = variable
        elif variable["type"] == "w" and variable["popNameShort"] == "PCL-PCL":
            wPC_PC = variable

    # Colores a usar en las representaciones y el stream de instantes de tiempo que ha durado la simulación
    colors = ["red", "green", "blue", "orange", "pink", "goldenrod"]
    timeStream = utils.generate_time_streams(data["simTime"], data["timeStep"], False)


    # Representamos V y spikes de cada PC
    """
    vMin = data["neuronParameters"][vPC["popNameShort"]]["v_rest"]
    vMax = data["neuronParameters"][vPC["popNameShort"]]["v_thresh"]
    utils.plot_spike_v_comb_all(vPC["data"], spikesPC["data"], [vMin, vMax], 1, data["simTime"], data["timeStep"],
                                vPC["popName"], False, True, plot, save, saveName + "_v_spike_" + vPC["popNameShort"], savePath)
    """

    # Representamos los spikes recibidos y emitidos por cada PC
    utils.plot_spike_pc_dg(spikesPC["data"], spikesDG["data"], timeStream, colors, 0.01, "Spikes DG-CA3", True, plot,
                           save, saveName+ "_spikes_DG_CA3", savePath)

    """
    # Representación de la evolución de los pesos de las sinapsis
    utils.plot_weight_single_neuron(wPC_PC["data"]["srcNeuronId"], wPC_PC["data"]["dstNeuronId"],
                                    wPC_PC["data"]["timeStamp"], wPC_PC["data"]["w"],
                                    [data["synParameters"]["PCL-PCL"]["w_min"]-0.5, data["synParameters"]["PCL-PCL"]["w_max"]+0.5], colors, plot, save, saveName,
                                    savePath)
    
    # Creamos una carpeta para los pesos de sinapsis individuales
    savePath = utils.check_folder(savePath + "synapseW/")
    if not savePath:
        print("Error al crear la carpeta de almacenamiento")
        return False
    
    # Representamos la evolución de los pesos de las sinapsis de forma individual
    utils.plot_weight_single_synapse(wPC_PC["data"]["srcNeuronId"], wPC_PC["data"]["dstNeuronId"],
                                    wPC_PC["data"]["timeStamp"], wPC_PC["data"]["w"],
                                    [data["synParameters"]["PCL-PCL"]["w_min"]-0.5, data["synParameters"]["PCL-PCL"]["w_max"]+0.5],
                                    plot, save, saveName, savePath)
    """
    return True


def main(plot, save, savePath, execute, fullPathFile, saveName):
    """
    Ejecuta el modelo o no y hace el plot indicado de los parámetros

    :param plot: bool que indica si se quiere o no mostrar los plots generados
    :param save: bool que indica si se quiere o no guardar dichos plots
    :param savePath: path donde almacenar los ficheros
    :param execute: si ejecutar o no la simulación del modelo
    :param fullPathFile: path + filename al archivo con los datos de la simulación en caso de que execute = False
    :param saveName: nombre base con el que almacenar los plots en caso de que execute = False
    :return:
    """
    # Ejecutamos el modelo
    if execute:
        fullPathFile, filename = CA3_oscilatory.main()
        saveName = filename
    # Hacemos el plot de los datos
    custom_plots(fullPathFile, plot, save, saveName, savePath)


if __name__ == "__main__":
    # Parámetros para modelar la ejecución y plot de los resultados
    # + Si representar la gráfica de las variables o no
    plot = False
    # + Si almacenar en imagen las gráficas
    save = True
    # + Dirección a la carpeta raíz donde almacenar los plots
    savePath = "plot/"
    # + Si ejecutar o no el modelo
    execute = True
    # + En caso de no ejecutar el modelo, se debe de especificar el path+filename al archivo con los datos a representar
    #    y el nombre base con el que almacenar los plots (recomendable que sea el nombre del fichero de datos)
    fullPathFile = "data/CA3_simple_2021_11_18__12_42_34.txt"
    saveName = "CA3_simple_2021_11_18__12_42_34"

    # Simulación y/o representación
    main(plot, save, savePath, execute, fullPathFile, saveName)
