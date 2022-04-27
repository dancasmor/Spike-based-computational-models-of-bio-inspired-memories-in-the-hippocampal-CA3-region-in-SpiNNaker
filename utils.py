
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
# Generación de datos estándar
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
# Plots default (para ejemplo lab manual)
#####################################

def plot_standar_stream(xStream, yStream, title, xlabel, ylabel, ylim, allValue, dotFig, lineDotFig, rotateLabels,
                        plot, saveFig, saveName, savePath, customXLabels=[]):
    """
    Toma todos los parámetros necesarios para poder crear, representar y/o guardar una gráfica de forma estándar

    :param xStream: stream de datos que usar en el eje x
    :param yStream: stream de datos que usar en el eje y
    :param title: título de la gráfica
    :param xlabel: etiqueta de valores del eje x
    :param ylabel: etiqueta de valores del eje y
    :param ylim: lista con los valores min y max a representar en el eje y -> [min, max]
    :param allValue: bool que indica si representar en los ejes los valores exactos de entrada (True) o aleatorios
                    entre el máximo y el mínimo
    :param dotFig: bool que indica si representar la figura como puntos exactos no conectados
    :param lineDotFig: bool que indica si representar la figura como un conjunto de puntos con líneas verticales
                    a la base
    :param rotateLabels: indica si rotar o no (90º) las etiquetas para que se puedan ver bien
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :param customXLabels: conjunto de etiquetas que usar en la gráfica para el eje x
    :return:
    """
    plt.figure()
    # Indicamos si queremos una representación contínua, discreta como puntos (scatterplot) o con puntos y líneas
    if lineDotFig:
        plt.stem(xStream, yStream, basefmt=" ")
    elif dotFig:
        plt.plot(xStream, yStream, 'o')
    else:
        plt.plot(xStream, yStream)
    # Añadimos los metadatos de texto
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(ylim)
    # Indicamos si queremos o no que aparezcan en los ejes únicamente los valores de entrada o aleatorios
    #  entre los valores max y min del stream
    if allValue:
        if not(customXLabels == []):
            plt.xticks(customXLabels)
        else:
            plt.xticks(xStream)
        plt.yticks(yStream)
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


def plot_default_spikes(spikes, spikeLim, simTime, timeStep, popName, plot, saveFig, saveName, savePath):
    """
    Representación con parámetros por defecto de spikes por neurona

    :param spikes: stream de spikes a representar
    :param spikeLim: [min, max] valores máximo y mínimo para representar los spikes en las gráficas
    :param simTime: tiempo de simulación
    :param timeStep: paso de tiempo usado en la simulación
    :param popName: nombre de la población de neuronas a la que pertenecen los spikes
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return:
    """
    # Generamos el stream de marcas temporales
    timeStream = generate_time_streams(simTime, timeStep, False)
    # Creamos la gráficas
    for indexNeuron, spike_neuron in enumerate(spikes):
        # Por cada neurona, hemos de marcar en que instantes de tiempo se producen los spikes
        spikesInTimeStep = [0] * len(timeStream)
        for indexTime, step in enumerate(timeStream):
            if step in spike_neuron:
                spikesInTimeStep[indexTime] = 1
            else:
                spikesInTimeStep[indexTime] = 0
        # Representamos los spikes a lo largo del tiempo de simulación para dicha neurona
        title = "Spikes: " + popName + " - neuronId " + str(indexNeuron)
        plot_standar_stream(timeStream, spikesInTimeStep, title, "Tiempo (ms)", "Spikes", spikeLim,
                            allValue=True, dotFig=True, rotateLabels=True, lineDotFig=True, plot=plot, saveFig=saveFig,
                            saveName=saveName + "_" + str(indexNeuron), savePath=savePath)


def plot_default_v(v, vLim, simTime, timeStep, popName, plot, saveFig, saveName, savePath):
    """
    Representación con parámetros por defecto de v por neurona

    :param v: stream de v a representar
    :param vLim: lista de valores mínimo y máximo que tendrá v -> [min, max]
    :param simTime: tiempo de simulación
    :param timeStep: paso de tiempo usado en la simulación
    :param popName: nombre de la población de neuronas a la que pertenecen los spikes
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return:
    """
    # Generamos el stream de tiempo
    timeStream = generate_time_streams(simTime, timeStep, False)
    # Representamos V
    for indexNeuron, v_neuron in enumerate(v):
        title = "Potencial de membrana: " + popName + " - neuronId " + str(indexNeuron)
        plot_standar_stream(timeStream, v_neuron, title, "Tiempo (ms)", "Potencial de membrana (uV)", vLim,
                            allValue=False, rotateLabels=False, dotFig=False, lineDotFig=False, plot=plot,
                            saveFig=saveFig, saveName=saveName + "_" + str(indexNeuron), savePath=savePath)


def plot_default_weight(x, y, z, zlimit, xlimit, colors, xlabel, ylabel, zlabel, title, plot, saveFig, saveName, savePath):
    """
    Representación de gráficos de la evolución de los pesos de la sinapsis. Por cada valor único del eje x se representa
    de forma continua los valores de "y" y "z" (normalmente x es srcNeuronId, y es el timeStamp y z es el peso)

    :param x: streams de datos del eje x
    :param y: streams de datos del eje y
    :param z: streams de datos del eje z
    :param zlimit: lista con 2 elementos, los valores mínimo y máximo del stream del eje z
    :param xlimit: lista con 2 elementos, los valores mínimo y máximo del stream del eje x
    :param colors: lista de colores para representar las diferentes neuronas
    :param xlabel: etiqueta de valores del eje x
    :param ylabel: etiqueta de valores del eje y
    :param zlabel: etiqueta de valores del eje z
    :param title: título de la gráfica
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return:
    """
    # Creación de la figura 3d
    fig = plt.figure(figsize=(19, 12))
    ax = plt.axes(projection="3d")

    # Representamos para cada valor único de x (cada id de neurona origen) la evolución de z(peso) en y(tiempo)
    xUniqueValues = list(set(x))
    for xUniqueValue in xUniqueValues:
        # Extraemos todos los índices que pertenecen a dicho valor de x
        xIndeces = [i for i, value in enumerate(x) if value == xUniqueValue]
        # Conseguimos los valores de dicha lista de índices de los demás ejes
        xForSameX = itemgetter(*xIndeces)(x)
        yForSameX = itemgetter(*xIndeces)(y)
        zForSameX = itemgetter(*xIndeces)(z)
        # Representamos
        ax.plot(xForSameX, yForSameX, zForSameX, color=colors[xUniqueValue], label="Neuron ID " + str(xUniqueValue))

    # Añadimos metadatos
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_zlabel(zlabel, fontsize=15)
    ax.set_xticks(xUniqueValues)
    # ax.set_yticks(y)
    ax.set_zlim3d(zlimit[0], zlimit[1])
    ax.set_xlim3d(xlimit[0], xlimit[1])
    ax.legend(fontsize=15)
    # ax.view_init(0, 0)

    # Guardamos y/o mostramos la figura
    if saveFig:
        plt.savefig(savePath + saveName + ".png")
    if plot:
        plt.show()
    plt.close()


#####################################
# Plots customs (para red CA3)
#####################################

def plot_spike_v_comb(v, spikes, timeStream, unit, vLim, marginLim, title, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera una gráfica en la que se representan los spikes y potencial de membrana de una neurona en el tiempo

    :param v: stream de v a representar
    :param spikes: stream de instantes de tiempo donde se ha generado un spike
    :param timeStream: stream de instantes de tiempos de la simulación
    :param unit: unidad del stream de tiempo
    :param vLim: lista de valores mínimo y máximo que tendrá v -> [min, max]
    :param marginLim: cantidad adicional (float) que se añade a los límites para evitar que se corten los bordes
    :param title: título de la gráfica
    :param rotateLabels: indica si rotar o no (90º) las etiquetas para que se puedan ver bien
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return: path+filename donde se ha almacenado la figura
    """

    # Instanciamos la figura
    plt.figure()

    # En caso de ocurrencia de spike, tomamos como potencial el valor anterior con el fin de corregir la gráfica de v
    for spike in spikes:
        v[timeStream.index(spike)] = vLim[0]

    # Añadimos el potencial como una línea contínua y líneas verticales donde se ha generado un spike
    plt.vlines(spikes, ymin=vLim[0] - marginLim, ymax=vLim[1] + marginLim, color="r", alpha=0.25, label="Spike")
    plt.plot(timeStream, v, label="v (uV)")
    plt.vlines(spikes, ymin=vLim[0], ymax=vLim[1])

    # Añadimos los metadatos de texto
    plt.xlabel("Tiempo de simulación (" + unit + ")")
    plt.ylabel("Potencial de membrana y spikes generados")
    plt.title(title)
    plt.ylim([vLim[0]-marginLim, vLim[1]+marginLim])
    plt.xticks(spikes)
    plt.legend()
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


def plot_spike_v_comb_all(v, spikes, vLim, marginLim, simTime, timeStep, popName, inSeg, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera por cada neurona una gráfica combinada del potencial con los spikes generados

    :param v: stream de v a representar
    :param spikes: stream de instantes de tiempo donde se ha generado un spike
    :param vLim: lista de valores mínimo y máximo que tendrá v -> [min, max]
    :param marginLim: cantidad adicional (float) que se añade a los límites para evitar que se corten los bordes
    :param simTime: tiempo de duración de la simulación
    :param timeStep: paso de tiempo de la simulación
    :param popName: nombre de la población a la que pertenecen los datos
    :param inSeg: si se quiere representar en segundos o ms
    :param rotateLabels: indica si rotar o no (90º) las etiquetas para que se puedan ver bien
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return:
    """
    # Generamos el stream de tiempo
    timeStream = generate_time_streams(simTime, timeStep, inSeg)
    unit = "ms"
    if inSeg:
        unit = "s"

    # Representación de los spikes y potencial de membrana
    for indexNeuron, v_neuron in enumerate(v):
        title = "Potencial de membrana y spikes: " + popName + " - neuronId " + str(indexNeuron)
        plot_spike_v_comb(v_neuron, spikes[indexNeuron], timeStream, unit, vLim, marginLim, title, rotateLabels,
                          plot, saveFig, saveName + "_" + str(indexNeuron), savePath)


def plot_spike_pre_post(spikesPre, spikePost, spikeInput, timeStream, indexPreNeurons, indexPostNeuron, colors, marginLim, title, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera una gráfica en la que se representan los spikes de una neurona postsináptica y todos los spikes presinápticos recibidos

    :param spikesPre: lista de streams (un stream por neurona presináptica que acabe en la misma neurona postsináptica)
                    de instantes de tiempo donde se ha generado un spike
    :param spikePost: stream de instantes de tiempo donde se ha generado un spike postináptico
    :param spikeInput: spikes generados por DG
    :param timeStream: stream de instantes de tiempos de la simulación
    :param indexPreNeurons: lista de índices de las neuronas postsinápticas
    :param indexPostNeuron: índice de la neurona postsináptica
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
    plt.figure(figsize=(25, 14))

    # Añadimos los spikes postsinápticos
    plt.vlines(spikePost, ymin=0-marginLim, ymax=1+marginLim, color=colors[indexPostNeuron],
               label="Post id " + str(indexPostNeuron) + " CA3")

    # Añadimos los spikes presinápticos
    # Por cada neurona presináptica que acaba en la neurona postsináptica de CA3
    for indexPreNeuron, spikePre in enumerate(spikesPre):
        plt.vlines(spikePre, ymin=0, ymax=1, color=colors[indexPreNeurons[indexPreNeuron]],
                        label="Pre id " + str(indexPreNeurons[indexPreNeuron]) + " CA3", alpha=0.3)
    # Idem pero por la neurona de DG que le corresponde
    plt.vlines(spikeInput, ymin=0, ymax=1, color="r", label="Pre DG")

    # Anotamos con números los spikes en cada instante de tiempo
    for stamp in timeStream:
        # Comprobamos si se ha generado un spike post o de entrada de DG
        label = ""
        if stamp in spikePost:
            label = label + str(indexPostNeuron)
        if stamp in spikeInput:
            label = label + ",DG"
        # Comprobación de si se ha generado un spike pre
        for indexPreNeuron, spikePre in enumerate(spikesPre):
            if stamp in spikePre:
                label = label + "," + str(indexPreNeurons[indexPreNeuron])
        # Realizamos la anotación sobre el instante temporal actual
        plt.annotate(label, xy=(stamp+0.1, 0.01), rotation=90, fontsize=15)

    # Añadimos los metadatos de texto
    plt.xlabel("Tiempo de simulación (ms)", fontsize=15)
    plt.ylabel("Spikes", fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylim([-marginLim, 1 + marginLim])
    plt.xlim(-0.5, max(timeStream) + 1.5)
    # Definimos la lista de marcas en el eje X
    listXticks = list(set(spikePost.copy() + [0, max(timeStream)+1] + [spike for sublist in spikesPre for spike in sublist] \
                 + spikeInput))
    # Añadimos las marcas
    plt.xticks(listXticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=13)
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


def plot_spike_pre_post_all(spikes, spikesInput, simTime, timeStep, colors, marginLim, popName, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera una gráfica en la que se representan los spikes de una neurona postsináptica y todos los spikes presinápticos recibidos

    :param spikes: lista de spikes de la población de CA3
    :param spikesInput: lista de spikes de DG
    :param simTime: tiempo de duración de la simulación
    :param timeStep: paso de tiempo de la simulación
    :param colors: lista de colores para representar las diferentes neuronas
    :param marginLim: cantidad adicional (float) que se añade a los límites para evitar que se corten los bordes
    :param popName: nombre de la población a la que pertenece las neuronas
    :param rotateLabels: indica si rotar o no (90º) las etiquetas para que se puedan ver bien
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return:
    """
    # Generamos el stream de tiempo
    timeStream = generate_time_streams(simTime, timeStep, False)

    # Creamos las gráficas por cada neurona
    for index, spike in enumerate(spikes):
        # Generamos la lista de spikes pre como la lista de todos menos el post
        spikesPre = spikes.copy()
        spikesPre.pop(index)
        # Generamos los índices pre como todos menos el post
        indexPreNeurons = list(range(0,len(spikes)))
        indexPreNeurons.remove(index)
        # Hacemos la representación de la gráfica
        title = "Spikes pre y postsinápticos: " + popName + " - neuronPostId " + str(index)
        plot_spike_pre_post(spikesPre=spikesPre, spikePost=spike, spikeInput=spikesInput[index], timeStream=timeStream,
                            indexPreNeurons=indexPreNeurons, indexPostNeuron=index,
                            colors=colors, marginLim=marginLim, title=title, rotateLabels=rotateLabels,
                            plot=plot, saveFig=saveFig, saveName=saveName + "_" + str(index), savePath=savePath)


def plot_weight_single_neuron(srcNeuronIds, dstNeuronIds, timeStreams, weights, zlimit, colors, plot, saveFig, saveName, savePath):
    """
    Crea gráficas 3d para representar por cada neurona única postsináptica, la evolución de pesos de las sinapsis que le
    llegan (neuronas presinápticas) a lo largo del tiempo

    :param srcNeuronIds: stream de neuronas presináptica
    :param dstNeuronIds: stream de neuronas postsinápticas
    :param timeStreams: stream de marcas temporales
    :param weights: stream de pesos de las sinapsis
    :param zlimit: límite de valores a representar en el eje Z, es decir, de pesos
    :param colors: lista de colores para representar las diferentes neuronas
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :return:
    """
    # Por cada neurona (única) destino de una sinapsis...
    dstUniqueIds = list(set(dstNeuronIds))
    for dstNeuronId in dstUniqueIds:
        # Extraemos todos los índices que pertenecen a dicho valor de id de neurona destino
        dstIndeces = [i for i, value in enumerate(dstNeuronIds) if value == dstNeuronId]
        # Conseguimos los valores de dicha lista de índices en los demás streams (neurona origen, time stamp y peso)
        srcForSameDst = itemgetter(*dstIndeces)(srcNeuronIds)
        timeForSameDst = itemgetter(*dstIndeces)(timeStreams)
        wForSameDst = itemgetter(*dstIndeces)(weights)

        # Los representamos
        plot_default_weight(x=srcForSameDst, y=timeForSameDst, z=wForSameDst, zlimit=zlimit, xlimit=[0, len(dstUniqueIds)],
                            colors=colors, xlabel="Src Neuron", ylabel="Tiempo (ms)", zlabel="Peso sinapsis (nA)",
                            title="Evolución peso PCL_i-PCL_" + str(dstNeuronId),
                            plot=plot, saveFig=saveFig, saveName=saveName + "_w_PCi_PC" + str(dstNeuronId),
                            savePath=savePath)


def plot_weight_single_synapse(srcNeuronIds, dstNeuronIds, timeStreams, weights, wlimit, plot, saveFig, saveName, savePath, debug=False, isSame=False):
    """
    Genera por cada sinapsis la evolución de su peso en el tiempo

    :param srcNeuronIds: stream de neuronas presináptica
    :param dstNeuronIds: stream de neuronas postsinápticas
    :param timeStreams: stream de marcas temporales
    :param weights: stream de pesos de las sinapsis
    :param wlimit: límite de valores a representar en el eje Z, es decir, de pesos
    :param plot: bool que indica si representar o no los plots generados
    :param saveFig: bool que indica si se quiere o no guardar dichos plots
    :param saveName: nombre base con el que almacenar los plots
    :param savePath: path donde almacenar los ficheros
    :param debug: bool que indica si imprimir información de depuración o no (cambio de pesos)
    :param isSame: bool que indica si existen conexiones donde origen y destino son iguales
    :return:
    """

    # Encontramos los ids únicos de neurona origen y destino
    srcUniqueIds = list(set(srcNeuronIds))
    dstUniqueIds = list(set(dstNeuronIds))

    # Representamos la evolución para cada sinapsis de forma independiente
    indexOrder = 0
    for srcNeuronId in srcUniqueIds:
        for dstNeuronId in dstUniqueIds:
            indexOrder = srcNeuronId + dstNeuronId*len(srcUniqueIds)
            # Excluimos, si procede, las sinapsis con uno mismo
            if srcNeuronId == dstNeuronId and not isSame:
                continue
            # Extraemos todos los índices que pertenecen a dicho par
            synapseIndeces = [i for i, value in enumerate(srcNeuronIds) if (value == srcNeuronId and dstNeuronIds[i] == dstNeuronId)]
            # Extraemos los valores para dicha sinapsis del stream de tiempo y peso
            timeSynapse = itemgetter(*synapseIndeces)(timeStreams)
            wSynapse = itemgetter(*synapseIndeces)(weights)

            # Buscamos los timestamps en los que se producen cambios de peso
            changeWTimeStamps = []
            if debug:
                print("Sinapsis " + str(srcNeuronId) + "->" + str(dstNeuronId))
            for i,w in enumerate(wSynapse):
                # Ignoramos la última iteración
                if i >= len(wSynapse)-1:
                    continue
                # Comprobamos si hay un cambio respecto a la siguiente iteración
                if not(w == wSynapse[i+1]):
                    # En caso afirmativo, añadimos el instante anterior al cambio
                    changeWTimeStamps.append(timeSynapse[i])
                    if debug:
                        print("(w="+str(w)+"->"+str(wSynapse[i+1])+" , t="+str(timeSynapse[i])+"->"+str(timeSynapse[i+1])+")")
            # Creamos la gráfica
            title = "Evolución peso sinapsis PC" + str(srcNeuronId) + "-PC" + str(dstNeuronId)
            plot_standar_stream(xStream=timeSynapse, yStream=wSynapse, title=title, xlabel="Tiempo (ms)",
                                ylabel="Peso (nA)", ylim=wlimit, allValue=True, dotFig=False, lineDotFig=True,
                                rotateLabels=True, plot=plot, saveFig=saveFig,
                                saveName=saveName + "_w_" + str(indexOrder) + "PC" + str(srcNeuronId) + "_PC" + str(dstNeuronId),
                                savePath=savePath, customXLabels=changeWTimeStamps)

def plot_spike_pre_post_pc_separate(spikesPCdir, spikesPCcont, spikesDG, timeStream, colors, marginLim, title, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera una gráfica en la que se representan los spikes recibidos y emitidos por las neuronas PC tanto dir como cont

    :param spikesPCdir: spikes generados por PCdir
    :param spikesPCcont: spikes generados por PCcont
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
    plt.figure(figsize=(25, 14))

    # Añadimos los spikes PC dir, cont y DG
    label = "DG"
    for spikeDG in spikesDG:
        plt.vlines(spikeDG, ymin=0, ymax=1, color=colors[2], label=label)
        label = "_nolegend_"
    label = "PCdir"
    for spikePCdir in spikesPCdir:
        plt.vlines(spikePCdir, ymin=0 - marginLim, ymax=1, color=colors[0], label=label)
        label = "_nolegend_"
    label = "PCcont"
    for spikePCcont in spikesPCcont:
        plt.vlines(spikePCcont, ymin=0, ymax=1 + marginLim, color=colors[1], label=label)
        label = "_nolegend_"

    # Construimos etiquetas en la que indicamos que neuronas han sido las que han generado spikes en cada instante
    for stamp in timeStream:
        label = ""
        # Comprobación de si se ha generado un spike PCdir, cont o DG
        for indexNeuron, spikeDG in enumerate(spikesDG):
            if stamp in spikeDG:
                label = label + "DG" + str(indexNeuron) + ","
        for indexNeuron, spikePCdir in enumerate(spikesPCdir):
            if stamp in spikePCdir:
                label = label + "PCd" + str(indexNeuron) + ","
        for indexNeuron, spikePCcont in enumerate(spikesPCcont):
            if stamp in spikePCcont:
                label = label + "PCc" + str(indexNeuron) + ","
        # Realizamos la anotación sobre el instante temporal actual
        plt.annotate(label, xy=(stamp+0.1, 0.01), rotation=90, fontsize=15)

    # Añadimos los metadatos de texto
    plt.xlabel("Tiempo de simulación (ms)", fontsize=15)
    plt.ylabel("Spikes", fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylim([-marginLim, 1 + marginLim])
    plt.xlim(-0.5, max(timeStream) + 1.5)
    # Definimos la lista de marcas en el eje X
    listXticks = list(set([spike for sublist in spikesPCdir for spike in sublist] + [0, max(timeStream)+1] + [spike for sublist in spikesPCcont for spike in sublist] \
                 + [spike for sublist in spikesDG for spike in sublist]))
    # Añadimos las marcas
    plt.xticks(listXticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=13)
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


def plot_all_spike_pre_post_decoder(spikesPCdir, spikesPCcont, spikesDG, spikesInput, numDirBinNeuron, numContNeuron, littleEndian, timeStream, colors, marginLim, title, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera una gráfica en la que se representan los spikes recibidos y emitidos por todas las neuronas

    :param spikesPCdir: spikes generados por PCdir
    :param spikesPCcont: spikes generados por PCcont
    :param spikesDG: spikes generados por DG (direcciones one-shot)
    :param spikesInput: spikes de entrada de la red (dirección y contenido binario)
    :param numDirBinNeuron: número de neuronas correspondiente a dirección en el array de entrada
    :param numContNeuron: número de neuronas usadas para almacenar el contenido
    :param littleEndian: si la dirección de entrada es little endian (true) o big endian (false)
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
    plt.figure(figsize=(25, 14))

    # Añadimos los spikes PC dir, cont, DG e Input
    label = "IN"
    for spikeIN in spikesInput:
        plt.vlines(spikeIN, ymin=0, ymax=1, color=colors[3], label=label)
        label = "_nolegend_"
    label = "DG"
    for indexNeuron, spikeDG in enumerate(spikesDG):
        if indexNeuron == 0:
            continue
        plt.vlines(spikeDG, ymin=0, ymax=1, color=colors[2], label=label)
        label = "_nolegend_"
    label = "PCdir"
    for spikePCdir in spikesPCdir:
        plt.vlines(spikePCdir, ymin=0 - marginLim, ymax=1, color=colors[0], label=label)
        label = "_nolegend_"
    label = "PCcont"
    for spikePCcont in spikesPCcont:
        plt.vlines(spikePCcont, ymin=0, ymax=1 + marginLim, color=colors[1], label=label)
        label = "_nolegend_"

    # Construimos etiquetas en la que indicamos que neuronas han sido las que han generado spikes en cada instante
    for stamp in timeStream:
        label = ""
        # Comprobación de si se ha generado un spike PCdir, cont, DG o IN
        # IN
        sublabeld = "INd"
        sublabelc = "INc"
        for indexNeuron, spikeIN in enumerate(spikesInput):
            if stamp in spikeIN:
                if indexNeuron < numDirBinNeuron:
                    label = label + sublabeld + str(indexNeuron)
                    sublabeld = "-"
                else:
                    label = label + sublabelc + str(indexNeuron)
                    sublabelc = "-"
        # DG
        sublabel = "DG"
        for indexNeuron, spikeDG in enumerate(spikesDG):
            if indexNeuron == 0:
                continue
            if stamp in spikeDG:
                label = label + sublabel + str(indexNeuron)
                sublabel = "-"
        # PCdir
        sublabel = "PCd"
        for indexNeuron, spikePCdir in enumerate(spikesPCdir):
            if stamp in spikePCdir:
                label = label + sublabel + str(indexNeuron)
                sublabel = "-"
        # PCcont
        sublabel = "PCc"
        for indexNeuron, spikePCcont in enumerate(spikesPCcont):
            if stamp in spikePCcont:
                label = label + sublabel + str(indexNeuron)
                sublabel = "-"
        # Realizamos la anotación sobre el instante temporal actual
        plt.annotate(label, xy=(stamp + 0.1, 0.01), rotation=90, fontsize=15)

    # Añadimos los metadatos de texto
    plt.xlabel("Tiempo de simulación (ms)", fontsize=15)
    plt.ylabel("Spikes", fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylim([-marginLim, 1 + marginLim])
    plt.xlim(-0.5, max(timeStream) + 1.5)
    # Definimos la lista de marcas en el eje X
    listXticks = list(
        set([spike for sublist in spikesPCdir for spike in sublist] + [0, max(timeStream) + 1] + [spike for sublist in
                                                                                                  spikesPCcont for spike
                                                                                                  in sublist] \
            + [spike for sublist in spikesDG for spike in sublist] + [spike for sublist in spikesInput for spike in sublist]))
    # Añadimos las marcas
    plt.xticks(listXticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=13)
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


def plot_in_out_spike_pre_post_decoder(spikesPCdir, spikesPCcont, spikesInput, numDirBinNeuron, numContNeuron, littleEndian, timeStream, colors, marginLim, title, rotateLabels, plot, saveFig, saveName, savePath):
    """
    Genera una gráfica en la que se representan los spikes recibidos y emitidos por todas las neuronas

    :param spikesPCdir: spikes generados por PCdir
    :param spikesPCcont: spikes generados por PCcont
    :param spikesInput: spikes de entrada de la red (dirección y contenido binario)
    :para, numDirBinNeuron: número de neuronas correspondiente a dirección en el array de entrada
    :param numContNeuron: número de neuronas usadas para almacenar el contenido
    :param littleEndian: si la dirección de entrada es little endian (true) o big endian (false)
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
    plt.figure(figsize=(19, 12))

    # Añadimos los spikes PC dir, cont, DG e Input
    label = "IN"
    for spikeIN in spikesInput:
        plt.vlines(spikeIN, ymin=0, ymax=1, color=colors[3], label=label)
        label = "_nolegend_"
    label = "PCdir"
    for spikePCdir in spikesPCdir:
        plt.vlines(spikePCdir, ymin=0 - marginLim, ymax=1, color=colors[0], label=label)
        label = "_nolegend_"
    label = "PCcont"
    for spikePCcont in spikesPCcont:
        plt.vlines(spikePCcont, ymin=0, ymax=1 + marginLim, color=colors[1], label=label)
        label = "_nolegend_"

    # Construimos etiquetas en la que indicamos que neuronas han sido las que han generado spikes en cada instante
    for stamp in timeStream:
        label = ""
        # Comprobación de si se ha generado un spike PCdir, cont o IN
        # IN
        sublabeld = "INd"
        sublabelc = "INc"
        for indexNeuron, spikeIN in enumerate(spikesInput):
            if stamp in spikeIN:
                if indexNeuron < numDirBinNeuron:
                    label = label + sublabeld + str(indexNeuron)
                    sublabeld = "-"
                else:
                    label = label + sublabelc + str(indexNeuron)
                    sublabelc = "-"
        # PCdir
        sublabel = "PCd"
        for indexNeuron, spikePCdir in enumerate(spikesPCdir):
            if stamp in spikePCdir:
                label = label + sublabel + str(indexNeuron)
                sublabel = "-"
        # PCcont
        sublabel = "PCc"
        for indexNeuron, spikePCcont in enumerate(spikesPCcont):
            if stamp in spikePCcont:
                label = label + sublabel + str(indexNeuron)
                sublabel = "-"
        # Realizamos la anotación sobre el instante temporal actual
        plt.annotate(label, xy=(stamp + 0.1, 0.01), rotation=90, fontsize=15)

    # Añadimos los metadatos de texto
    plt.xlabel("Tiempo de simulación (ms)", fontsize=15)
    plt.ylabel("Spikes", fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylim([-marginLim, 1 + marginLim])
    plt.xlim(-0.5, max(timeStream) + 1.5)
    # Definimos la lista de marcas en el eje X
    listXticks = list(
        set([spike for sublist in spikesPCdir for spike in sublist] + [0, max(timeStream) + 1] + [spike for sublist in
                                                                                                  spikesPCcont for spike
                                                                                                  in sublist] \
            + [spike for sublist in spikesInput for spike in sublist]))
    # Añadimos las marcas
    plt.xticks(listXticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=13)
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


def represent_in_out_data_format(spikesPCdir, spikesPCcont, spikesInput, spikesDG, numDirBinNeuron, numDirOneHot, numContNeuron, littleEndian, timeStream, saveName, savePath):
    """
        Genera un txt con la información de entrada y salida en un formato más representativo (representación en números,
        no spikes)

        :param spikesPCdir: spikes generados por PCdir
        :param spikesPCcont: spikes generados por PCcont
        :param spikesInput: spikes de entrada de la red (dirección y contenido binario)
        :param spikesDG: spikes generados por DG (direcciones one-shot)
        :param numDirBinNeuron: número de neuronas correspondiente a dirección en el array de entrada
        :param numDirOneHot: número de neuronas correspondiente a dirección en one hot
        :param numContNeuron: número de neuronas usadas para almacenar el contenido
        :param littleEndian: si la dirección de entrada es little endian (true) o big endian (false)
        :param timeStream: stream de instantes de tiempos de la simulación
        :param saveName: nombre base con el que almacenar los plots
        :param savePath: path donde almacenar los ficheros
        :return: path+filename donde se ha almacenado la figura, si procede
    """

    # Diccionarios que almacenarán toda la información intermedia y el de salida formateada
    allRawInfo, allFormatInfo = {}, {}

    # Creamos un array con los índices de las posiciones del vector IN para ajustarlo al tipo de codificación
    if littleEndian:
        indexInputSpike = np.flip(range(numDirBinNeuron)).tolist() + list(range(numContNeuron))
    else:
        indexInputSpike = list(range(numDirBinNeuron)) + list(range(numContNeuron))

    # Comprobación de que neuronas han generado spikes en cada instante y los almacenamos ordenadamente en un diccionario
    for stamp in timeStream:
        # Spikes IN
        listINDirSpikes = []
        listINContSpikes = []
        for indexNeuron, spikeIN in enumerate(spikesInput):
            if stamp in spikeIN:
                if indexNeuron < numDirBinNeuron:
                    listINDirSpikes.append(indexInputSpike[indexNeuron])
                else:
                    listINContSpikes.append(indexInputSpike[indexNeuron])
        # Spikes DG
        listDGSpikes = []
        for indexNeuron, spikeDG in enumerate(spikesDG):
            if indexNeuron == 0:
                continue
            if stamp in spikeDG:
                listDGSpikes.append(indexNeuron)
        # Spikes PC
        listPCDirSpikes = []
        listPCContSpikes = []
        for indexNeuron, spikePCdir in enumerate(spikesPCdir):
            if stamp in spikePCdir:
                listPCDirSpikes.append(indexNeuron)
        for indexNeuron, spikePCcont in enumerate(spikesPCcont):
            if stamp in spikePCcont:
                listPCContSpikes.append(indexNeuron)
        # Comprobamos si hay algún spike en dicho time stamp
        hasSpike = listINDirSpikes or listINContSpikes or listDGSpikes or listPCDirSpikes or listPCContSpikes

        # Los almacenamos en el diccionario usando el time stamp
        allRawInfo.update({stamp:{"hasSpike":hasSpike, "INdir":listINDirSpikes, "INcont":listINContSpikes, "DG":listDGSpikes,
                                     "PCdir":listPCDirSpikes, "PCcont":listPCContSpikes}})

    # Almacenamos el fichero con los datos raw
    write_file(savePath, saveName + "_raw", allRawInfo)

    # Tomamos cada instante y lo formateamos
    indexInputSpike = range(numDirBinNeuron)
    if littleEndian:
        indexInputSpike = np.flip(indexInputSpike).tolist()
    for stamp, spikes in allRawInfo.items():
        # + Inputs dir -> convertir de binario a decimal en función de la codificación
        formatINdir = False
        #   1) Extraemos la lista de spikes
        spikesINdir = spikes["INdir"]
        #   2) Comprobamos si está vacío
        if spikesINdir:
            #   3) Convertimos de binario a decimal
            formatINdir = 0
            for spikeINdir in spikesINdir:
                formatINdir = formatINdir + pow(2, indexInputSpike[spikeINdir])

        # + Input cont -> dejamos en binario
        formatINcont = spikes["INcont"]

        # + DG -> dejamos en binario (restamos 1 para trabajar de 1 a n)
        formatDG = False
        spikesDG = spikes["DG"]
        if spikesDG:
            formatDG = spikesDG[0] - 1

        # + PCdir -> añadimos 1 al índice actual, en lugar de interpretar el índice del spike como el índice en el que se
        #           coloca el 1 en one-hot, lo podemos tomar como la posición real de 1 a n
        formatPCdir = False
        spikesPCdir = spikes["PCdir"]
        if spikesPCdir:
            formatPCdir = spikesPCdir[0]+1

        # + PCcont -> mantenemos en binario
        formatPCcont = spikes["PCcont"]

        # Comprobamos si hay algún spike en dicho time stamp
        hasSpike = formatINdir is not False or formatINcont or formatDG is not False or\
                   formatPCdir is not False or formatPCcont

        # Solo almacenamos si hay información
        if hasSpike:
            # Los almacenamos en el diccionario usando el time stamp
            allFormatInfo.update(
                {stamp: {"INdir": formatINdir, "INcont": formatINcont, "DG": formatDG,
                         "PCdir": formatPCdir, "PCcont": formatPCcont}})


    # Almacenamos el fichero con los datos formateados en formato json/diccionario
    write_file(savePath, saveName + "_format", allFormatInfo)

    # Formateo a tipo cadena de texto
    allSpikePrintFormat = ""
    # Tomamos cada instante
    for stamp, spikes in allFormatInfo.items():
        allSpikePrintFormat = allSpikePrintFormat + "+ t = " + str(stamp) + " ms: \n"
        # + Inputs dir
        if spikes["INdir"] is not False:
            allSpikePrintFormat = allSpikePrintFormat + "\t * INdir (decimal) = " + str(spikes["INdir"]) + "\n"

        # + Input cont
        if spikes["INcont"]:
            allSpikePrintFormat = allSpikePrintFormat + "\t * INcont (binario) = " + str(spikes["INcont"]) + "\n"

        # + DG
        if spikes["DG"] is not False:
            allSpikePrintFormat = allSpikePrintFormat + "\t * DG (one hot) = "
            dgOneHot = ""
            for index in range(numDirOneHot):
                if index == spikes["DG"]:
                    dgOneHot = dgOneHot + "1"
                else:
                    dgOneHot = dgOneHot + "0"
            allSpikePrintFormat = allSpikePrintFormat + dgOneHot[::-1] + "\n"

        # + PCdir
        if spikes["PCdir"] is not False:
            allSpikePrintFormat = allSpikePrintFormat + "\t * PCdir (decimal) = " + str(spikes["PCdir"]) + "\n"

        # + PCcont -> mantenemos en binario
        if spikes["PCcont"]:
            allSpikePrintFormat = allSpikePrintFormat + "\t * PCcont (binario) = " + str(spikes["PCcont"]) + "\n"

        allSpikePrintFormat = allSpikePrintFormat + "\n"

    # Almacenamos el fichero con los datos formateados en formato de cadena de texto
    write_file(savePath, saveName + "_print_format", allSpikePrintFormat)

    # Creamos el formato de tabla numDirBinNeuron, numDirOneHot, numContNeuron
    tableFormatInfo = "\t {:<15} {:<15} {:<35} {:<15} {:<15} {:<35} {:<15} \n" \
                      "-----------------------------------------------------------" \
                      "-----------------------------------------------------------" \
                      "------------------------------------------------------\n".format("TimeStamp (ms)", "INdir (decimal)",
                                                                                 "INcont", "DG (one-hot)",
                                                                                 "PCdir (decimal)", "PCcont", "Operación")
    for stamp, spikes in allFormatInfo.items():
        # Convertimos los False y [] en -, y reformateamos aquellos campos que sean necesarios

        # INdir
        inDir = spikes["INdir"] if not (spikes["INdir"] == []) and (spikes["INdir"] is not False) else "-"

        # INcont -> representar tanto las posiciones a 1 como la representación binaria
        if not (spikes["INcont"] == []) and (spikes["INcont"] is not False):
            inContBin = ""
            for index in range(numContNeuron):
                if index in spikes["INcont"]:
                    inContBin = "1" + inContBin
                else:
                    inContBin = "0" + inContBin
            inContBin = inContBin + " " + str(spikes["INcont"])
        else:
            inContBin = "-"

        # DG -> representación one hot
        if not (spikes["DG"] == []) and (spikes["DG"] is not False):
            dgOneHot = ""
            for index in range(numDirOneHot):
                if index == spikes["DG"]:
                    dgOneHot = dgOneHot + "1"
                else:
                    dgOneHot = dgOneHot + "0"
        else:
            dgOneHot = "-"
        dgOneHot = dgOneHot[::-1]

        # PCdir
        pcDir = spikes["PCdir"] if not (spikes["PCdir"] == []) and (spikes["PCdir"] is not False) else "-"

        # PCcont -> representar tanto las posiciones a 1 como la representación binaria
        if not (spikes["PCcont"] == []) and (spikes["PCcont"] is not False):
            pcContBin = ""
            for index in range(numContNeuron):
                if index in spikes["PCcont"]:
                    pcContBin = "1" + pcContBin
                else:
                    pcContBin = "0" + pcContBin
            pcContBin = pcContBin + " " + str(spikes["PCcont"])
        else:
            pcContBin = "-"

        # Comprobamos el tipo de operación que se está realizando si se ha comenzado
        operationName = "-"
        if not(inDir == "-") and not(inContBin == "-"):
            operationName = "Escritura (Aprendizaje)"
        elif not(inDir == "-"):
            operationName = "Lectura (Recordar)"

        # Elaboramos las filas de la tabla
        tableFormatInfo = tableFormatInfo + "\t {:<15} {:<15} {:<35} {:<15} {:<15} {:<35}  {:<15} \n" \
                      "-----------------------------------------------------------" \
                      "-----------------------------------------------------------" \
                      "------------------------------------------------------\n".format(stamp,
                                                                                                     str(inDir),
                                                                                                     str(inContBin),
                                                                                                     str(dgOneHot),
                                                                                                     str(pcDir),
                                                                                                     str(pcContBin),
                                                                                                     operationName)

    # Almacenamos el fichero con los datos formateados en formato de tabla
    write_file(savePath, saveName + "_print_format_table", tableFormatInfo)

