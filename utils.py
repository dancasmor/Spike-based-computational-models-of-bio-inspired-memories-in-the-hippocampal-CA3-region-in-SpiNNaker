
import time
import matplotlib.pyplot as plt
from operator import itemgetter
import os
import numpy as np


#####################################
# Input/Output
#####################################

def write_file(basePath, title, data):
    """
    Escribe en un fichero los datos de entrada

    :param basePath: path en el que almacenar el fichero de datos
    :param title: título del fichero a crear
    :param data: los datos que se quieren almacenar en el fichero
    :return: fullPath (path + filename), filename (nombre del fichero creado sin extensión ni path)
    """
    # Extraemos la fecha y hora actuales
    strDate = time.strftime("%Y_%m_%d__%H_%M_%S")
    # Creamos el nombre del fichero como año_mes_dia_hora.txt
    filename = title + "_" + strDate
    fullPath = basePath + filename + ".txt"
    # Escribimos los datos en el fichero
    file = open(fullPath, "w")
    file.write(str(data))
    file.close()
    # Devolvemos el path al archivo guardado
    return fullPath, filename


def read_file(fullPath):
    """
    Lee el fichero pasado por parámetro

    :param fullPath: path + filename del fichero a leer
    :return: los datos leidos del fichero o False si no se ha podido acceder al mismo
    """
    try:
        file = open(fullPath, "r")
        return eval(file.read())
    except FileNotFoundError:
        return False


def check_folder(path):
    """
    Comprueba si existe una carpeta y, en caso de que no exista, la crea

    :param path: path a donde se encuentra la carpeta a comprobar/crear
    :return: el path si se ha creado/existe o False si no existe y, además, no se ha podido crear
    """
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
            return path
        except OSError as e:
            print("Error al crear el directorio")
            return False
    else:
        return path


#####################################
# Formateo de datos
#####################################

def format_neo_data(tipo, stream, timeStream={}):
    """
    Formatea los streams de datos generados en neo para eliminar las cabeceras

    :param tipo: tipo de dato que se está manejando ("v", "spikes" y "weights" soportados actualmente)
    :param stream: el stream de datos (en formato neo salvo el del peso) que se quiere formatear
    :param timeStream: parámetro opcional usado para el formateo del stream del peso -> {"simTime", "timeStep"}
    :return: el stream formateado
    """
    if timeStream is None:
        timeStream = []
    if tipo == "v":
        formatStream = format_v_stream(stream)
    elif tipo == "spikes":
        formatStream = format_spike_stream(stream)
    elif tipo == "weights":
        formatStream = format_weight_stream(stream, timeStream)
    else:
        formatStream = False
    return formatStream


def format_v_stream(vStream):
    """
    Cambia el formato de streams neo para el potencial de membrana a otro más sencillo sin cabeceras

    :param vStream: el stream de potenciales de membrana en formato neo
    :return: el stream de v formateado
    """
    formatV = []
    # Obtenemos la matriz de valores ->
    #   cada elemento representa un time_stamp y el contenido de dicho elemento el valor para cada neurona
    rawStream = vStream.as_array().tolist()
    # Comprobamos cuantas neuronas hay en cada time_stamp
    numNeurons = len(rawStream[0])
    # Reformateamos para que se al revés, cada elemento sea una neurona y el contenido los valores para cada time_stamp
    for neuron in range(0, numNeurons):
        formatV.append([item[neuron] for item in rawStream])
    # Comprobamos que no haya valores con "nan"
    for indexNeuron, neuron in enumerate(formatV):
        for indexTime, timeStamp in enumerate(neuron):
            if str(formatV[indexNeuron][indexTime]) == "nan":
                if indexTime == 0:
                    formatV[indexNeuron][indexTime] = -60.0
                elif indexTime >= len(neuron)-1:
                    formatV[indexNeuron][indexTime] = formatV[indexNeuron][indexTime-1]
                else:
                    formatV[indexNeuron][indexTime] = (formatV[indexNeuron][indexTime-1] +
                                                      formatV[indexNeuron][indexTime+1])/2
    return formatV


def format_spike_stream(spikes):
    """
    Cambia el formato de streams neo para los spikes a otro más sencillo sin cabeceras

    :param spikes: el stream de spikes en formato neo
    :return: el stream de spikes formateado
    """
    formatSpikes = []
    # Extraemos los spikes para cada neurona
    for neuron in spikes:
        formatSpikes.append(neuron.as_array().tolist())
    return formatSpikes


def format_weight_stream(weights, timeParam):
    """
    Devuelve la información de los pesos separadas en diferentes campos/variables listo para ser representados

    :param weights: el stream con todas las variables mezcladas relacionadas con el peso y la sinapsis a la que pertenece
    :param timeParam: parámetros temporales de la simulación -> {"simTime", "timeStep"}
    :return: los parámetros del stream de pesos divididos en 4 variables -> {"srcNeuronId", "dstNeuronId", "w", "timeStamp"}
    """
    srcNeuronId, dstNeuronId, w, timeStampStream = [], [], [], []

    # Generamos la secuencia temporal en ms
    timeStream = generate_time_streams(timeParam["simTime"], timeParam["timeStep"], False, True)

    # Por cada marca temporal de la simulación...
    for indexStep, step in enumerate(weights):
        # Para cada sinapsis en dicha marca temporal...
        for indexSyn, synapse in enumerate(step):
            """
            # En caso de que haya tuplas corruptas
            if not (type(synapse) == tuple):
                continue
            """
            # Extraemos los valores de las tuplas y las almacenados de forma separada
            srcNeuronId.append(synapse[0])
            dstNeuronId.append(synapse[1])
            w.append(synapse[2])
            timeStampStream.append(timeStream[indexStep])
    return {"srcNeuronId": srcNeuronId, "dstNeuronId": dstNeuronId, "w": w, "timeStamp": timeStampStream}


def get_last_stamp_synapse_list(dataPath, delay=1.0, synapse="PCL-PCL"):
    """
    Dado el path+filename del fichero de datos de una prueba, extraemos el peso de la última iteración

    :param dataPath: path+filename del fichero con los datos de una prueba anterior
    :param delay: delay que añadir a la sinapsis
    :param synapse: nombre de la sinapsis del que extraer los pesos
    :return: lista de sinapsis (src,dst,w) del último timestamp y otra lista con metadatos de las conexiones
    """

    # Abrimos el archivo con los datos de los pesos
    data = read_file(dataPath)

    # Buscamos la variable de peso de las sinapsis PCL-PCL
    w = {}
    for variable in data["variables"]:
        if variable["type"] == "w" and variable["popNameShort"] == synapse:
            w = variable["data"]

    # Comprobamos que hemos encontrado los datos deseados
    if w == {}:
        print("Error al leer el archivo")
        return False

    # Seleccionamos el último timeStamp (el más grande)
    maxTimeStamp = max(list(set(w["timeStamp"])))

    # Extraemos todos los índices que pertenecen a dicho valor de timestamp
    maxTimeStampIndeces = [i for i, value in enumerate(w["timeStamp"]) if value == maxTimeStamp]

    # Conseguimos los valores de peso, id origen y destino para dichos timeStamps
    lastTimeStampWeights = itemgetter(*maxTimeStampIndeces)(w["w"])
    lastTimeStampSrcNeuron = itemgetter(*maxTimeStampIndeces)(w["srcNeuronId"])
    lastTimeStampDstNeuron = itemgetter(*maxTimeStampIndeces)(w["dstNeuronId"])

    # Creamos la lista de tuplas
    synapses = []
    for index, w_individual in enumerate(lastTimeStampWeights):
        synapses.append((lastTimeStampSrcNeuron[index], lastTimeStampDstNeuron[index], w_individual, delay))

    # Devolvemos la lista de sinapsis y la metainformación de la sinapsis original STDP
    return synapses, data["synParameters"][synapse]


#####################################
# Data generation
#####################################

def generate_time_streams(simTime, timeStep, timeInSeg, endPlus=False):
    """
    Genera una secuencia temporal en s o ms de la duración de la simulación usando el timestep de la misma

    :param simTime: tiempo de simulación de la red (ms)
    :param timeStep: paso de tiempo usado en la simulación (ms)
    :param timeInSeg: bool que indica si se quiere el stream en segundo o ms
    :param endPlus: bool opcional que indica si se quiere incluir el valor de simTime dentro de la secuencia o no
    :return: la secuencia temporal configurada por los parámetros de entrada
    """
    # Tomar el instante de tiempo final de la secuencia o no
    if endPlus:
        endCount = 1
    else:
        endCount = 0

    # Generar la secuencia en s o ms
    if timeInSeg:
        timeStream = generate_sequence(0, simTime + endCount, timeStep, 1000)
    else:
        timeStream = generate_sequence(0, simTime + endCount, timeStep, 1)
    return timeStream


def generate_sequence(start, stop, step, divisor):
    """
    Genera una secuencia de números dada las condiciones de entrada

    :param start: comienzo de la secuencia
    :param stop: parada de la secuencia
    :param step: incremento en cada iteración -> tanto int como float
    :param divisor: valor por el que dividir la cuenta a la hora de almacenarla
    :return: la secuencia de valores generados
    """
    sequence = []
    count = start
    while count < stop:
        sequence.append(float(count)/divisor)
        count = count + step
    return sequence


#####################################
# Plot
#####################################


def plot_spike_pc_dg(spikesPC, spikesDG, timeStream, colors, marginLim, title, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera una gráfica en la que se representan los spikes recibidos y emitidos por las neuronas PC y DG

    :param spikesPCdir: spikes generados por PCd
    :param spikesDG: spikes generados por DG
    :param timeStream: stream de instantes de tiempos de la simulación
    :param colors: lista de colores para representar las diferentes neuronas
    :param marginLim: cantidad adicional (float) que se añade a los límites para evitar que se corten los bordes
    :param title: título de la gráfica
    :param rotateLabels: indica si rotar o no (90º) las etiquetas para que se puedan ver bien
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return: path+filename donde se ha almacenado la figura, si procede
    """

    # Instanciamos la figura
    plt.figure(figsize=(24, 8))

    # Añadimos los spikes PC dir, cont y DG
    label = "DG"
    for spikeDG in spikesDG:
        plt.vlines(spikeDG, ymin=0 - marginLim, ymax=0.5 , color=colors[0], label=label)
        label = "_nolegend_"
    label = "PC"
    for spikePC in spikesPC:
        plt.vlines(spikePC, ymin=0, ymax=0.5 + marginLim, color=colors[1], label=label)
        label = "_nolegend_"

    # Construimos etiquetas en la que indicamos que neuronas han sido las que han generado spikes en cada instante
    for stamp in timeStream:
        label = ""
        # Comprobación de si se ha generado un spike PC o DG
        sublabel = "DG"
        for indexNeuron, spikeDG in enumerate(spikesDG):
            if stamp in spikeDG:
                label = label + sublabel + str(indexNeuron)
                sublabel = "-"
        label = label + " "
        sublabel = "PC"
        for indexNeuron, spikePC in enumerate(spikesPC):
            if stamp in spikePC:
                label = label + sublabel + str(indexNeuron)
                sublabel = "-"
        # Realizamos la anotación sobre el instante temporal actual
        plt.annotate(label, xy=(stamp+0.1, 0.01), rotation=90, fontsize=15)

    # Añadimos los metadatos de texto
    plt.xlabel("Simulation time (ms)", fontsize=15)
    plt.ylabel("Spikes", fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylim([-marginLim, 0.5 + marginLim])
    plt.xlim(-0.5, max(timeStream) + 1.5)
    # Definimos la lista de marcas en el eje X
    listXticks = list(set([spike for sublist in spikesPC for spike in sublist] \
                 + [spike for sublist in spikesDG for spike in sublist]))
    # Añadimos las marcas
    plt.xticks(listXticks, fontsize=15)
    # plt.yticks(fontsize=15)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=15)
    # Rotar o no las etiquetas
    if rotateLabels:
        plt.xticks(rotation=90)

    # Guardamos y/o mostramos la figura
    if saveFig:
        plt.savefig(savePath + saveName + ".png")
    if plot:
        plt.show()
    plt.close()
    # Devolvemos la dirección a donde se ha almacenado la figura, si procede
    return savePath + saveName + ".png"
