import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def show_values(pc, fmt="%.4f", **kw):
    """
    Shows corresponding values of cells inside the cell on the heatmap
    :param pc: the plot
    :param fmt: format for numbers to be displayed
    :param kw: allow additional arguments for text formatting
    """
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    """
    Allows to specify figure size in centimeter in matplotlib
    :param tupl: accepts both ((value1, value2)) and (value1, value2)
    :return: a tuple of values in centimeters
    """
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
            correct_orientation=False, cmap='RdBu'):
    """
    Creates a heat map, given the parameters from the classification report
    :param AUC: Area Under the Curve (AUC) from prediction scores
    :param title: title of the graph
    :param xlabel: label to be displayed at the x axis: Metrics (e.g. Precision, Recall)
    :param ylabel: label to be displayed at the y axis: Classes (e.g. positive, negative)
    :param xticklabels: tick labels for x axis
    :param yticklabels: tick labels for y axis
    :param figure_width: width of the figure in cm
    :param figure_height: height of figure in cm
    :param correct_orientation: possibility to invert the y axis
    :param cmap: color scheme
    """
    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    # labels = [xlabel, ylabel]
    # ax.legend(labels)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, name_of_model, cmap='RdBu'):
    """
    Create a plot for a classification report as returned by sklearn.metrics.classification_report
    :param classification_report: a string, the classification report to be plotted
    :param name_of_model: name of the model to be written on the picture title
    :param cmap: color scheme for the hear map
    """

    title = 'Classification Report for ' + name_of_model
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 1)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)


    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)


def create_classification_report_plot(report, results_folder, name_of_model):
    """
    Given a classification report, create a heat map for precision, recall and f1 score for each sentiment label
    :param report: classification report for a model as returned by sklearn.metrics.classification_report
    :param results_folder: name of folder where results for report are stored
    :param name_of_model: name of parameters of the model, same scheme as names of .txt files in "results" folders
    """

    report = report.replace('micro avg', 'micro_avg')
    report = report.replace('macro avg', 'macro_avg')
    report = report.replace('weighted avg', 'weighted_avg')
    #report = report.partition('micro avg')[0]

    save_path = os.path.join(Path(__file__).parents[2].__str__(), results_folder, '{}.png'.format(name_of_model))
    plot_classification_report(report, name_of_model)
    plt.savefig(save_path, dpi=200, format='png', bbox_inches='tight')
    plt.close()
