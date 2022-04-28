
import random
import utils
import CA3_pc_inhibitory
import CA3_pc_inhibitory_static_syn


def custom_plots(fullPathFile, plot, save, saveName, savePath):
    """
    Processing the data from a simulation to get a visual representation of the result

    :param fullPathFile: the full path to the file with the data recorded from the simulation
    :param plot: if show the plot in running time
    :param save: if store the plots
    :param saveName: the base name used to store the generated files
    :param savePath: the base path used to store the generated files
    :return: True if the simulation and/or the creation of the visual representation of the data has been done correctly
            or False in other cases
    """
    # Open data file of the simulation
    data = utils.read_file(fullPathFile)
    if not data:
        print("Error to open data file")
        return False
    # Create folder to store all the generated files
    savePath = utils.check_folder(savePath + saveName + "/")
    if not savePath:
        print("Error to create a folder to store generated files")
        return False

    # Search all variables which are going to be used to create the plots and representations
    vPC, spikesPC, spikesDG, wPC_PC, vINH, spikesINH = {}, {}, {}, {}, {}, {}
    for variable in data["variables"]:
        if variable["type"] == "spikes" and variable["popNameShort"] == "PCL":
            spikesPC = variable
        elif variable["type"] == "spikes" and variable["popNameShort"] == "DGL":
            spikesDG = variable

    # Create the stream of time stamp and color to use in representations
    colors = ["red", "green", "blue", "orange", "pink", "goldenrod"]
    timeStream = utils.generate_time_streams(data["simTime"], data["timeStep"], False)


    # Create a spike plot of all activations of DG and PC (CA3) neurons
    utils.plot_spike_pc_dg(spikesPC["data"], spikesDG["data"], timeStream, colors, 0.01, "Spikes DG-CA3", True, plot,
                           save, saveName + "_spikes_DG_CA3", savePath)

    return True


def main(plot, save, savePath, execute, executeSTDPCA3, fullPathFile, saveName):
    """
    Execute the simulation of the network and/or create a visual representation of the data recorded

    :param plot: if show the plot in running time
    :param save: if store the plots
    :param savePath: the base path used to store the generated files
    :param execute: if execute or not the simulation, in case of false, a fullPathFile is needed
    :param executeSTDPCA3: if execute dinamic or static version
    :param fullPathFile: the full path to the file with the data recorded from the simulation
    :param saveName: the base name used to store the generated files
    :return:
    """
    # Execute the model if applicable
    if execute:
        if executeSTDPCA3:
            fullPathFile, filename = CA3_pc_inhibitory.main()
            saveName = filename
        else:
            fullPathFile, filename = CA3_pc_inhibitory_static_syn.main()
            saveName = filename
    # Processing the data and plot it
    custom_plots(fullPathFile, plot, save, saveName, savePath)


if __name__ == "__main__":
    # Plot and execution parameters
    # + If show the plot in running time
    plot = False
    # + If store the plots
    save = True
    # + Base path where store the plot
    savePath = "plot/"
    # + If execute the network or take already generated data
    execute = True
    # + if execute dinamic or static version
    executeSTDPCA3 = False
    # + If not execute, the full path to the file with the data recorded from the simulation and the base name used to store
    #   the generated files (txt, png, ...)
    fullPathFile = "data/CA3_simple_2021_11_18__12_42_34.txt"
    saveName = "CA3_simple_2021_11_18__12_42_34"

    # Simulation and/or representation
    main(plot, save, savePath, execute, executeSTDPCA3, fullPathFile, saveName)
