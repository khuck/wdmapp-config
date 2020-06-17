#!/usr/bin/env python3
import matplotlib
matplotlib.use('svg')
import pandas as pd
from mpi4py import MPI
import numpy as np
import adios2
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('fast')
import operator
from operator import add
from matplotlib.font_manager import FontProperties
import json
import io
from pathlib import Path
import os
import sys
if sys.version_info[0] < 3 or sys.version_info[1] < 3:
    raise Exception("Must be using Python 3.3 or newer.")

host_bbox = 'tight'
rank_bbox = 'tight'
top_x_bbox = 'tight'

def SetupArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instream", "-i", help="Name of the input stream", required=True)
    parser.add_argument("--config", "-c", help="Name of the config JSON file", default="charts.json")
    parser.add_argument("--nompi", "-nompi", help="ADIOS was installed without MPI", action="store_true")
    args = parser.parse_args()

    return args

def get_num_hosts(attr_info):
    names = {}
    # Iterate over the metadata, and get our hostnames.
    # Build a dictionary of unique values, if the value is
    # already there overwrite it.
    for key in attr_info:
        if "Hostname" in key:
            names[(attr_info[key]['Value'])] = 1
    return len(names)

def get_valid_ranks(attr_info):
    ranks_per_host = {}
    for key in attr_info:
        if "Hostname" in key:
            rank_id = int(key.split(':')[1])
            host_name = attr_info[key]['Value']
            if host_name in ranks_per_host:
                ranks_per_host[host_name].append(rank_id)
            else:
                ranks_per_host[host_name] = [rank_id,]
    valid_ranks = [min(ranks_per_host[host]) for host in ranks_per_host]
    return valid_ranks

# Get the tight bbox once per figure because it is slow

def get_renderer_bbox(ax):
    fig = ax.get_figure()
    fig.canvas.print_svg(io.BytesIO())
    bbox = fig.get_tightbbox(fig._cachedRenderer).padded(0.35)
    return bbox

# Format the output image file name
def get_image_filename(config, fr_step, step):
    imgfile = ""
    if "filename" not in config:
        config["filename"] = config["name"].replace(" ", "-")
    if config["Timestep for filename"] == "default":
        imgfile = config["SVG output directory"]+"/"+config["filename"]+"_"+"{0:0>5}".format(step)+".svg"
    else:
        stepnumber = fr_step.read(config["Timestep for filename"])
        if (stepnumber.size == 0):
            stepnumber = 0
        else: 
            stepnumber = int(stepnumber[0])
        imgfile = config["SVG output directory"]+"/"+config["filename"]+"_"+"{0:0>5}".format(stepnumber)+".svg"
    
    return imgfile

# Build a dataframe that has per-node data for this timestep of the output data

def build_per_host_dataframe(fr_step, step, num_hosts, valid_ranks, config):
    # Read the number of ranks - check for the new method first
    num_ranks = 1
    if len(fr_step.read('num_ranks')) == 0:
        num_ranks = len(fr_step.read('num_threads'))
    else:
        num_ranks = fr_step.read('num_ranks')[0]

    # Find out how many ranks per node we have (ceiling division)
    ranks_per_node = (num_ranks // num_hosts) + 1
    rows = []
    # For each variable, get each MPI rank's data, some will be bogus (they didn't write it)
    for name in config["components"]:
        rows.append(fr_step.read(name))
    if len(rows[0]) == 0:
        print("No rows!  Is TAU configured correctly?")
        return
    print("Processing dataframe...")
    # Now, transpose the matrix so that the rows are each rank, and the variables are columns
    df = pd.DataFrame(rows).transpose()
    # Add a name for each column
    df.columns = config["labels"]
    # Add the MPI rank column (each row is unique)
    df['mpi_rank'] = range(0, len(df))
    # Add the step column, all with the same value
    df['step']=step
    # Filter out the rows that don't have valid data (keep only the lowest rank on each host)
    # This will filter out the bogus data
    df_trimmed = df[df['mpi_rank'].isin(valid_ranks)]
    print("Plotting...")
    ax = df_trimmed[config["labels"]].plot(kind='bar', stacked=True)
    # Create default axes labels if they're not configured
    if "x axis" not in config: 
        config["x axis"]="Node"
    if "y axis" not in config:
        config["y axis"]="Percent"
    ax.set_xlabel(config["x axis"])
    ax.set_ylabel(config["y axis"])
    plt.xticks(rotation='horizontal')
    if "legend columns" not in config:
        config["legend columns"] = 3
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.12), ncol=config["legend columns"])
    imgfile = get_image_filename(config, fr_step, step)
    print("Writing...")
    global host_bbox
    if step == 0:
        host_bbox = get_renderer_bbox(ax)
    plt.savefig(imgfile, bbox_inches=host_bbox)
    plt.close()
    print("done.")

# Build a dataframe that has per-rank data for this timestep of the output data

def build_per_rank_dataframe(fr_step, step, config):
    rows = []
    # For each variable, get each MPI rank's data
    for name in config["components"]:
        rows.append(fr_step.read(name))
    if len(rows[0]) == 0:
        print("No rows!  Is TAU configured correctly?")
        return
    print("Processing dataframe...")
    # Now, transpose the matrix so that the rows are each rank, and the variables are columns
    df = pd.DataFrame(rows).transpose()
    # Add a name for each column
    df.columns = config["labels"]
    # Add the MPI rank column (each row is unique)
    df['mpi_rank'] = range(0, len(df))
    # Add the step column, all with the same value
    df['step']=step
    print("Plotting...")
    ax = df[config["labels"]].plot(logy=True, style=config["plot style"])
    # Create default axes labels if they're not configured
    if "x axis" not in config:
        config["x axis"]="Rank"
    if "y axis" not in config:
        config["y axis"]="Value"
    ax.set_xlabel(config["x axis"])
    ax.set_ylabel(config["y axis"])
    if "legend columns" not in config:
        config["legend columns"] = 2
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.11), ncol=config['legend columns'])
    imgfile = get_image_filename(config, fr_step, step)
    print("Writing...")
    global rank_bbox
    if step == 0:
        rank_bbox = get_renderer_bbox(ax)
    plt.savefig(imgfile, bbox_inches=rank_bbox)
    plt.close()
    print("done.")

# Build a dataframe for the top X timers

def build_topX_timers_dataframe(fr_step, step, config):
    #variables = fr_step.get_variable_names()
    totalTime = fr_step.read('.TAU application / Inclusive TIME')[0]
    variables = fr_step.available_variables()
    num_threads = fr_step.read('num_threads')[0]
    timer_data = {}
    # Get all timers
    #for name, _ in variables:
    for name in variables:
        if ".TAU application" in name:
            continue
        if "addr=" in name:
            continue
        if "Exclusive TIME" in name:
            shortname = name.replace(" / Exclusive TIME", "")
            timer_data[shortname] = []
            temp_vals = fr_step.read(name)
            for i in temp_vals:
                if i > totalTime:
                    timer_data[shortname].append(0)
                else: 
                    timer_data[shortname].append(i)
    print("Processing dataframe...")
    df = pd.DataFrame(timer_data)
    # Get the mean of each column
    mean_series = df.mean()
    # Get top X timers
    sorted_series = mean_series.sort_values(ascending=False)
    topX = int(config["granularity"])
    topX_cols = sorted_series[:topX].axes[0].tolist()
    # Add all other timers together
    other_series = sorted_series[topX+1:].axes[0].tolist()
    df["other"] = 0
    for other_col in other_series: 
        df[other_col].clip(lower=0)
        df["other"] += df[other_col]
    topX_cols.insert(0,"other")
    # Plot the DataFrame
    print("Plotting...")
    ax = df[topX_cols].plot(kind='bar', stacked=True, width=1.0)
    # Create default axes labels if they're not configured
    if "x axis" not in config:
        config["x axis"] = "Rank"
    if "y axis" not in config:
        config["y axis"] = "Time"
    ax.set_xlabel(config["x axis"])
    ax.set_ylabel(config["y axis"])
    if "number of ticks" not in config:
        config["number of ticks"] = 8
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i%(config["number of ticks"]) !=0]
    handles, labels = ax.get_legend_handles_labels()
    if "max label length" not in config:
        config["max label length"] = 30 # default value if not set
    short_labels = [label[0:config["max label length"]] for label in labels]
    if "legend columns" not in config:
        config["legend columns"] = 2
    plt.legend(reversed(handles), reversed(short_labels), loc='upper center', bbox_to_anchor=(0.5,-0.12), ncol=config['legend columns'])
    imgfile = get_image_filename(config, fr_step, step)
    print("Writing...")
    global top_x_bbox
    if step == 0:
        top_x_bbox = get_renderer_bbox(ax)
    plt.savefig(imgfile, bbox_inches=top_x_bbox)
    plt.close()
    print("done.")


# Process the ADIOS2 file

def process_file(args):
    with open(args.config) as config_data:
        config = json.load(config_data)

    # make the output directory
    if "SVG output directory" not in config or config["SVG output directory"] == ".":
        config["SVG output directory"] = os.getcwd()
    else:
        Path(config["SVG output directory"]).mkdir(parents=True, exist_ok=True)
    
    if "Timestep for filename" not in config:
        config["Timestep for filename"] = "default"
    
    for f in config["figures"]:
        if "SVG output directory" not in f or f["SVG output directory"] == ".":
            f["SVG output directory"] = config["SVG output directory"]
        else:
            Path(config["SVG output directory"]).mkdir(parents=True, exist_ok=True)
        if "Timestep for filename" not in f:
             f["Timestep for filename"] = config["Timestep for filename"]
        
    filename = args.instream
    print ("Opening:", filename)
    if not args.nompi:
        fr = adios2.open(filename, "r", MPI.COMM_SELF, "adios2.xml", "TAUProfileOutput")
    else:
        fr = adios2.open(filename, "r", config["ADIOS2 config file"], "TAUProfileOutput")
    # Get the attributes (simple name/value pairs)
    attr_info = fr.available_attributes()
    # Get the unique host names from the attributes
    num_hosts = get_num_hosts(attr_info)
    cur_step = 0
    # Iterate over the steps
    for fr_step in fr:
        begin_time = time.time()
        # track current step
        cur_step = fr_step.current_step()
        print(filename, "Step = ", cur_step)
        for f in config["figures"]:
            print(f["name"])
            if "Timer" in f["name"]:
                build_topX_timers_dataframe(fr_step, cur_step, f)
            elif f["granularity"] == "node":
                valid_ranks = get_valid_ranks(attr_info)
                build_per_host_dataframe(fr_step, cur_step, num_hosts, valid_ranks, f)
            else:
                build_per_rank_dataframe(fr_step, cur_step, f)
        fr.end_step()
        total_time = time.time() - begin_time
        print(f"Processed step in {total_time} seconds", flush=True)


if __name__ == '__main__':
    args = SetupArgs()
    begin_time = time.time()
    process_file(args)
    total_time = time.time() - begin_time
    print(f"Processed file in {total_time} seconds", flush=True)
